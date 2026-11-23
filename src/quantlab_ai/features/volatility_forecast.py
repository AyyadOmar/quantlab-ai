"""Next-session open-to-close volatility estimated at the prior session's close."""
import numpy as np
import pandas as pd


VOLATILITY_COLUMNS = ["intraday_vol_forecast", "intraday_vol_delta_1d"]
BASELINE_COLUMN = "intraday_vol_baseline_20"
DECAY = 0.94
BASELINE_WINDOW = 20


def add_volatility_features(history, raw):
    """Add a daily RMS return forecast, its daily change, and a scoring baseline.

    The recursive variance starts at the first squared intraday return and puts
    6% weight on each new observation. Forecasts are in decimal return units.
    Only completed session open/close prices enter the calculation.
    """
    result = history.copy()
    prices = raw.copy()
    for frame in (result, prices):
        dates = pd.to_datetime(frame["date"])
        if frame.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Expected nonempty, unique, chronological daily prices")
        frame["date"] = dates
    prices = prices.loc[prices["date"] <= result["date"].max()].set_index("date")
    if prices.empty:
        raise ValueError("No completed price sessions through the last signal date")
    for frame in (result, prices):
        if "ticker" in frame and (frame["ticker"].isna().any() or frame["ticker"].nunique() != 1):
            raise ValueError("Supply one ticker per volatility history")
    if "ticker" in result and "ticker" in prices and result["ticker"].iat[0] != prices["ticker"].iat[0]:
        raise ValueError("Price and feature tickers disagree")
    values = prices[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Open and close prices must be finite and positive")
    aligned = prices.reindex(result["date"])
    if not np.allclose(aligned[["open", "close"]], result[["open", "close"]], rtol=1e-12, atol=0):
        raise ValueError("Missing prices or inconsistent price snapshots")

    squared_return = (prices["close"] / prices["open"] - 1).pow(2)
    variance = squared_return.ewm(alpha=1 - DECAY, adjust=False).mean()
    forecast = np.sqrt(variance)
    prices[VOLATILITY_COLUMNS[0]] = forecast
    prices[VOLATILITY_COLUMNS[1]] = forecast.diff()
    prices[BASELINE_COLUMN] = np.sqrt(squared_return.rolling(BASELINE_WINDOW).mean())
    columns = VOLATILITY_COLUMNS + [BASELINE_COLUMN]
    features = prices.loc[result["date"], columns].to_numpy()
    if not np.isfinite(features).all():
        raise ValueError("Insufficient price history for volatility forecasts on all signal dates")
    result[columns] = features
    return result


def _squared_variance_errors(scored_rows):
    columns = [VOLATILITY_COLUMNS[0], BASELINE_COLUMN, "next_day_return"]
    values = scored_rows[columns].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values[:, :2] < 0).any():
        raise ValueError("Every scored session needs two finite nonnegative forecasts and a finite return")
    variance = (100 * values) ** 2
    errors = variance[:, :2] - variance[:, 2, None]
    return errors ** 2


def score_volatility(scored_rows):
    """Score forecasts against next-session squared return, a noisy variance proxy.

    Compare mean squared variance errors on the same realized sessions. Squared
    returns are in percentage points squared, so MSE is in percentage points to
    the fourth power. These scores describe the size of returns, regardless of sign.
    """
    errors = _squared_variance_errors(scored_rows)
    ewma_mse, baseline_mse = np.mean(errors, axis=0)
    return {"rows": len(errors), "ewma_variance_mse": float(ewma_mse),
            "baseline_20_variance_mse": float(baseline_mse)}


def score_volatility_gain(scored_rows, seed=42):
    """Relative pooled MSE reduction and a descriptive paired circular block interval.

    Sampling contiguous sessions together retains cross-company and local temporal
    dependence. A relative reduction is undefined when baseline error is zero.
    """
    keys = scored_rows[["ticker", "execution_date"]].copy()
    keys["execution_date"] = pd.to_datetime(keys["execution_date"])
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError("Expected one score per ticker and execution date")
    keys[["ewma_error", "baseline_error"]] = _squared_variance_errors(scored_rows)
    daily = keys.groupby("execution_date")[["ewma_error", "baseline_error"]].sum().to_numpy()
    ewma_error, baseline_error = daily.sum(axis=0)
    sessions = len(daily)
    block = min(20, sessions)
    repetitions = 2000
    result = {"error_reduction": None, "interval_95": None, "sessions": sessions,
              "block_sessions": block, "repetitions": repetitions}
    if baseline_error == 0:
        return result
    result["error_reduction"] = float(1 - ewma_error / baseline_error)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, sessions, size=(sessions + block - 1) // block)
        sample = ((starts[:, None] + offsets) % sessions).ravel()[:sessions]
        model, reference = daily[sample].sum(axis=0)
        if reference == 0:
            return result
        draws.append(1 - model / reference)
    result["interval_95"] = np.quantile(draws, [.025, .975]).tolist()
    return result
