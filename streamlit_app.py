from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "docs" / "assets"
DEMO_DIR = ROOT / "docs" / "demo"


st.set_page_config(
    page_title="QuantLab AI",
    page_icon="📈",
    layout="wide",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


summary = load_json(DEMO_DIR / "aapl_xgboost_summary.json")
benchmarks = load_json(DEMO_DIR / "aapl_xgboost_benchmarks.json")

st.title("QuantLab AI")
st.caption("Quantitative machine learning research platform for stock prediction, walk-forward validation, and benchmark-aware backtesting.")

st.markdown(
    """
    QuantLab AI is a production-style Python application that studies whether ML-generated directional signals
    can survive realistic evaluation. Instead of presenting a one-shot backtest, the project uses expanding-window
    walk-forward validation and compares model signals against baseline strategies like buy-and-hold, always-long,
    and naive momentum.
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "XGBoost")
col2.metric("Ticker", "AAPL")
col3.metric("Strategy Return", format_pct(summary["total_return"]))
col4.metric("Buy & Hold", format_pct(summary["benchmark_return"]))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Benchmarks", "Visuals", "Architecture"])

with tab1:
    st.subheader("Why this project matters")
    st.markdown(
        """
        - Production-style project structure, not notebook-only experimentation
        - Multiple model families: Logistic Regression, Random Forest, XGBoost, and PyTorch LSTM
        - Technical and market-context features, including relative strength and rolling beta versus `SPY`
        - Time-aware validation to reduce misleading performance estimates
        - Benchmark-aware backtesting to test whether predictive signals create real trading value
        """
    )

    st.subheader("Key findings")
    st.markdown(
        """
        - Walk-forward validation reduced apparent performance versus earlier single-split experiments.
        - The current XGBoost baseline outperformed naive momentum, but underperformed passive exposure.
        - The platform shows that classification signal and tradable alpha are not the same thing.
        - The strongest value of the project is its research workflow, evaluation discipline, and extensibility.
        """
    )

with tab2:
    st.subheader("Walk-forward benchmark snapshot")

    benchmark_frame = pd.DataFrame(
        [
            {
                "Strategy": "Model Strategy",
                "Total Return": format_pct(summary["total_return"]),
                "Max Drawdown": format_pct(summary["max_drawdown"]),
                "Sharpe Ratio": f"{summary['sharpe_ratio']:.2f}",
                "Win Rate": format_pct(summary["win_rate"]),
                "Trade Count": summary["trade_count"],
            },
            {
                "Strategy": "Buy & Hold",
                "Total Return": format_pct(benchmarks["buy_and_hold"]["total_return"]),
                "Max Drawdown": format_pct(benchmarks["buy_and_hold"]["max_drawdown"]),
                "Sharpe Ratio": f"{benchmarks['buy_and_hold']['sharpe_ratio']:.2f}",
                "Win Rate": format_pct(benchmarks["buy_and_hold"]["win_rate"]),
                "Trade Count": benchmarks["buy_and_hold"]["trade_count"],
            },
            {
                "Strategy": "Always Long",
                "Total Return": format_pct(benchmarks["always_long"]["total_return"]),
                "Max Drawdown": format_pct(benchmarks["always_long"]["max_drawdown"]),
                "Sharpe Ratio": f"{benchmarks['always_long']['sharpe_ratio']:.2f}",
                "Win Rate": format_pct(benchmarks["always_long"]["win_rate"]),
                "Trade Count": benchmarks["always_long"]["trade_count"],
            },
            {
                "Strategy": "Momentum",
                "Total Return": format_pct(benchmarks["momentum"]["total_return"]),
                "Max Drawdown": format_pct(benchmarks["momentum"]["max_drawdown"]),
                "Sharpe Ratio": f"{benchmarks['momentum']['sharpe_ratio']:.2f}",
                "Win Rate": format_pct(benchmarks["momentum"]["win_rate"]),
                "Trade Count": benchmarks["momentum"]["trade_count"],
            },
        ]
    )
    st.dataframe(benchmark_frame, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Model visuals")
    left, right = st.columns(2)
    with left:
        st.image(str(ASSETS_DIR / "aapl_xgboost_equity_curve.png"), caption="AAPL XGBoost Equity Curve")
        st.image(str(ASSETS_DIR / "aapl_random_forest_confusion_matrix.png"), caption="Random Forest Confusion Matrix")
    with right:
        st.image(str(ASSETS_DIR / "aapl_lstm_equity_curve.png"), caption="AAPL LSTM Equity Curve")
        st.image(str(ASSETS_DIR / "aapl_candlestick.png"), caption="AAPL Candlestick Chart")

with tab4:
    st.subheader("System design")
    st.markdown(
        """
        1. Download target ticker data and `SPY` market context from Yahoo Finance.
        2. Engineer technical indicators and market-relative features.
        3. Train models using expanding-window walk-forward validation.
        4. Convert out-of-sample probabilities into trading signals.
        5. Backtest against benchmark strategies and export charts, metrics, and model artifacts.
        """
    )

    st.code(
        """quantlab-ai/
├── src/quantlab_ai/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── backtesting/
│   ├── visualization/
│   └── pipeline.py
├── docs/assets/
├── docs/demo/
└── streamlit_app.py""",
        language="text",
    )

st.divider()
st.markdown(
    "Live demo built with Streamlit. Source code and full project documentation are available in the GitHub repository."
)
