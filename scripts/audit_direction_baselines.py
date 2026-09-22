"""Post-hoc diagnostics of saved forecasts; does not fit or select any model."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "backtesting/recent_history_study_v8"


def attach_baselines(frame, history):
    history = history.sort_values("date").copy()
    if history["date"].duplicated().any():
        raise ValueError("Duplicate signal dates")
    # Same-day open/close is resolved at our after-close prediction cutoff.
    # Never use the target column here: it contains the NEXT session's outcome.
    history["recent_up_share"] = (history["close"] > history["open"]).rolling(20, min_periods=20).mean()
    result = frame.merge(history[["date", "recent_up_share"]], on="date", how="left", validate="one_to_one")
    if result["recent_up_share"].isna().any():
        raise ValueError("Missing rolling history")
    result["always_up"] = 1
    result["always_down"] = 0
    result["training_majority"] = (result["training_prevalence"] >= .5).astype(int)
    result["recent_majority"] = (result["recent_up_share"] >= .5).astype(int)
    return result


def score(frame, column):
    y, pred = frame.target, frame[column]
    down = y == 0
    up = y == 1
    return [len(frame), (pred == y).mean(), balanced_accuracy_score(y, pred),
            (pred[down] == 0).mean(), (pred[up] == 1).mean()]


def table(rows):
    text = ["| Comparison | Rows | Accuracy | Balanced accuracy | Down recall | Up recall |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, values in rows:
        text.append("| " + label + " | " + str(values[0]) + " | " + " | ".join(f"{v:.2%}" for v in values[1:]) + " |")
    return "\n".join(text)


def main():
    report = ["# Direction diagnostic audit", "",
              "These are post-hoc diagnostics of existing frozen predictions, not new untouched evidence. "
              "No model, cutoff, training window, or prospective forecast has changed. "
              "Always-down is a fixed diagnostic, not a policy selected for deployment using these outcomes. "
              "Rolling-majority uses the last 20 completed intraday outcomes available at each signal close; "
              "ties predict up. Training-majority uses saved training prevalence.", ""]
    for period in ["historical", "confirmation"]:
        frames = {}
        for model in ["logistic_regression", "xgboost"]:
            pieces = []
            for ticker in ["aapl", "msft", "nvda"]:
                prefix = "confirmation_" if period == "confirmation" else ""
                frame = pd.read_csv(STUDY / f"{prefix}{ticker}_{model}_validation_selected_predictions.csv")
                folder = ROOT / "data/processed/recent_history_study_v8"
                if period == "confirmation":
                    folder = folder / "confirmation"
                history = pd.read_csv(folder / f"{ticker.upper()}_features.csv")
                pieces.append(attach_baselines(frame, history))
            frames[model] = pd.concat(pieces, ignore_index=True)
        lr, xgb = frames.values()
        keys = ["ticker", "date", "execution_date"]
        left, right = lr.sort_values(keys), xgb.sort_values(keys)
        if not left[keys + ["target"]].reset_index(drop=True).equals(right[keys + ["target"]].reset_index(drop=True)):
            raise ValueError("Models must be compared on identical outcomes")
        report += [f"## {period.title()}: {lr.execution_date.min()} through {lr.execution_date.max()}", ""]
        rows = [("Logistic regression", score(lr, "prediction")), ("XGBoost", score(xgb, "prediction"))]
        for col in ["always_up", "always_down", "training_majority", "recent_majority"]:
            label = "Training Majority (LR)" if col == "training_majority" else col.replace("_", " ").title()
            rows.append((label, score(lr, col)))
        report += [table(rows), "", "### Logistic regression by company and time", ""]
        rows = [(ticker, score(group, "prediction")) for ticker, group in lr.groupby("ticker")]
        dates = np.sort(lr.execution_date.unique())
        midpoint = dates[len(dates) // 2]
        rows += [("First half", score(lr[lr.execution_date < midpoint], "prediction")),
                 ("Second half", score(lr[lr.execution_date >= midpoint], "prediction"))]
        report += [table(rows), ""]
        for model, frame in frames.items():
            auc = roc_auc_score(frame.target, frame.prob_up)
            brier = ((frame.prob_up - frame.target) ** 2).mean()
            reference = ((frame.training_prevalence - frame.target) ** 2).mean()
            report += [f"{model}: pooled ROC AUC {auc:.4f}; Brier {brier:.4f}; training-prevalence Brier {reference:.4f} (lower is better).", ""]
    report += ["## Interpretation", "",
               "The model's absolute accuracy and its margin over always-up answer different questions. "
               "A period with fewer up days makes always-up less accurate. The newer result is concentrated "
               "in NVIDIA; matching the baseline on other companies is not replication. "
               "Balanced accuracy weights up-day and down-day recall equally. "
               "These tables diagnose weaknesses; they do not justify choosing a new rule from the scored dates. "
               "No trading-return or profitability claim follows from these classification metrics.", ""]
    path = ROOT / "docs/direction_diagnostic_audit.md"
    path.write_text("\n".join(report))
    print(path)


if __name__ == "__main__":
    main()
