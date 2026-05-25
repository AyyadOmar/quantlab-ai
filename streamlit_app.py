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
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_ratio(value: float) -> str:
    return f"{value:.2f}"


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 24%),
                linear-gradient(180deg, #08111f 0%, #0c1728 42%, #0f1b2d 100%);
            color: #e5edf8;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }
        h1, h2, h3 {
            color: #f8fbff !important;
            letter-spacing: -0.02em;
        }
        p, li, div, span {
            color: #c7d3e3;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(11, 18, 32, 0.98));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        }
        [data-testid="stMetricLabel"] {
            color: #8ea2bc;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
        }
        [data-testid="stMetricValue"] {
            color: #f8fbff;
            font-size: 1.8rem;
        }
        .hero-shell {
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(30, 41, 59, 0.82), rgba(15, 23, 42, 0.96)),
                linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(16, 185, 129, 0.08));
            padding: 1.6rem 1.7rem 1.4rem 1.7rem;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.28);
        }
        .eyebrow {
            display: inline-block;
            color: #7dd3fc;
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            font-size: 3.1rem;
            line-height: 1.0;
            font-weight: 700;
            margin: 0 0 0.8rem 0;
            color: #f8fbff;
        }
        .hero-copy {
            max-width: 54rem;
            font-size: 1.03rem;
            line-height: 1.7;
            color: #c8d5e7;
            margin-bottom: 0.25rem;
        }
        .hero-callout {
            margin-top: 1rem;
            border-left: 3px solid #38bdf8;
            padding-left: 1rem;
            color: #d9e6f4;
            font-size: 0.98rem;
        }
        .section-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(10, 17, 30, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.20);
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.8rem;
        }
        .mini-card {
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.72);
            border-radius: 16px;
            padding: 0.95rem 1rem;
        }
        .mini-label {
            color: #8ea2bc;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .mini-value {
            color: #f8fbff;
            font-size: 1.28rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .positive {
            color: #4ade80 !important;
        }
        .negative {
            color: #f87171 !important;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }
        .pill {
            border: 1px solid rgba(125, 211, 252, 0.18);
            background: rgba(14, 116, 144, 0.12);
            color: #d7eefc;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            font-size: 0.86rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            overflow: hidden;
        }
        [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        button[data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            padding: 0.55rem 1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.32), rgba(14, 165, 233, 0.22));
            border-color: rgba(96, 165, 250, 0.38);
        }
        .footer-note {
            color: #94a3b8;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


summary = load_json(DEMO_DIR / "aapl_xgboost_summary.json")
benchmarks = load_json(DEMO_DIR / "aapl_xgboost_benchmarks.json")

benchmark_frame = pd.DataFrame(
    [
        {
            "Strategy": "Model Strategy",
            "Total Return": format_pct(summary["total_return"]),
            "Max Drawdown": format_pct(summary["max_drawdown"]),
            "Sharpe Ratio": format_ratio(summary["sharpe_ratio"]),
            "Win Rate": format_pct(summary["win_rate"]),
            "Trade Count": summary["trade_count"],
        },
        {
            "Strategy": "Buy & Hold",
            "Total Return": format_pct(benchmarks["buy_and_hold"]["total_return"]),
            "Max Drawdown": format_pct(benchmarks["buy_and_hold"]["max_drawdown"]),
            "Sharpe Ratio": format_ratio(benchmarks["buy_and_hold"]["sharpe_ratio"]),
            "Win Rate": format_pct(benchmarks["buy_and_hold"]["win_rate"]),
            "Trade Count": benchmarks["buy_and_hold"]["trade_count"],
        },
        {
            "Strategy": "Always Long",
            "Total Return": format_pct(benchmarks["always_long"]["total_return"]),
            "Max Drawdown": format_pct(benchmarks["always_long"]["max_drawdown"]),
            "Sharpe Ratio": format_ratio(benchmarks["always_long"]["sharpe_ratio"]),
            "Win Rate": format_pct(benchmarks["always_long"]["win_rate"]),
            "Trade Count": benchmarks["always_long"]["trade_count"],
        },
        {
            "Strategy": "Momentum",
            "Total Return": format_pct(benchmarks["momentum"]["total_return"]),
            "Max Drawdown": format_pct(benchmarks["momentum"]["max_drawdown"]),
            "Sharpe Ratio": format_ratio(benchmarks["momentum"]["sharpe_ratio"]),
            "Win Rate": format_pct(benchmarks["momentum"]["win_rate"]),
            "Trade Count": benchmarks["momentum"]["trade_count"],
        },
    ]
)


def value_class(value: float) -> str:
    return "positive" if value >= 0 else "negative"


st.markdown(
    f"""
    <div class="hero-shell">
        <div class="eyebrow">Quantitative Machine Learning Research Platform</div>
        <div class="hero-title">QuantLab AI</div>
        <div class="hero-copy">
            A production-style research system for testing whether stock-direction models generate
            signals that survive walk-forward validation, benchmark comparison, and realistic backtesting.
        </div>
        <div class="hero-callout">
            The core research premise is simple: <strong>classification signal and tradable edge are not the same thing</strong>.
            This platform is built to measure that gap honestly.
        </div>
        <div class="pill-row">
            <div class="pill">Walk-Forward Validation</div>
            <div class="pill">Benchmark-Aware Backtesting</div>
            <div class="pill">Technical + Market Context Features</div>
            <div class="pill">scikit-learn + XGBoost + PyTorch</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Ticker", "AAPL")
metric2.metric("Lead Model", "XGBoost")
metric3.metric("Strategy Return", format_pct(summary["total_return"]))
metric4.metric("Buy & Hold Return", format_pct(summary["benchmark_return"]))

tab_overview, tab_benchmarks, tab_visuals, tab_architecture = st.tabs(
    ["Research Overview", "Benchmarks", "Visual Analytics", "System Design"]
)

with tab_overview:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("What the platform does")
        st.markdown(
            """
            - Downloads historical ticker data and market context from Yahoo Finance
            - Engineers technical signals and relative-strength features versus `SPY`
            - Trains Logistic Regression, Random Forest, XGBoost, and LSTM models
            - Generates out-of-sample predictions using expanding-window walk-forward validation
            - Converts prediction probabilities into trade signals
            - Backtests the resulting strategy against simple but meaningful baselines
            """
        )
        st.subheader("Why this is a stronger project")
        st.markdown(
            """
            Most student stock-prediction repos stop at classification accuracy or a single backtest curve.
            QuantLab AI is designed as a research workflow: modular code, realistic evaluation, and explicit
            comparison against passive and naive baselines.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Latest benchmark snapshot")
        st.markdown(
            f"""
            <div class="mini-grid">
                <div class="mini-card">
                    <div class="mini-label">Strategy Return</div>
                    <div class="mini-value {value_class(summary["total_return"])}">{format_pct(summary["total_return"])}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Buy & Hold</div>
                    <div class="mini-value positive">{format_pct(summary["benchmark_return"])}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Max Drawdown</div>
                    <div class="mini-value negative">{format_pct(summary["max_drawdown"])}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Trade Count</div>
                    <div class="mini-value">{summary["trade_count"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **Research takeaway**

            The current baseline demonstrates why quant evaluation must go beyond directional prediction.
            A model can show some classification signal and still fail to produce robust trading performance.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab_benchmarks:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Walk-forward performance versus benchmark strategies")
    st.caption("Latest public benchmark snapshot for the AAPL XGBoost experiment.")
    st.dataframe(benchmark_frame, use_container_width=True, hide_index=True)

    bench_left, bench_right = st.columns(2, gap="large")
    with bench_left:
        st.markdown(
            """
            **What is being compared**

            - **Model Strategy**: probability-thresholded ML signals
            - **Buy & Hold**: passive exposure benchmark
            - **Always Long**: simplified constant participation baseline
            - **Momentum**: naive rule based on prior-day direction
            """
        )
    with bench_right:
        st.markdown(
            """
            **Interpretation**

            - The model beat naive momentum
            - It did not beat passive exposure
            - Drawdown and turnover remain too high
            - Further work should focus on thresholding, filtering, and feature robustness
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_visuals:
    st.subheader("Research visuals")
    image_left, image_right = st.columns(2, gap="large")
    with image_left:
        st.image(str(ASSETS_DIR / "aapl_xgboost_equity_curve.png"), caption="AAPL XGBoost Strategy vs Benchmarks")
        st.image(str(ASSETS_DIR / "aapl_random_forest_confusion_matrix.png"), caption="Random Forest Confusion Matrix")
    with image_right:
        st.image(str(ASSETS_DIR / "aapl_lstm_equity_curve.png"), caption="AAPL LSTM Strategy vs Benchmarks")
        st.image(str(ASSETS_DIR / "aapl_candlestick.png"), caption="AAPL Candlestick Chart")

with tab_architecture:
    arch_left, arch_right = st.columns([1.05, 0.95], gap="large")
    with arch_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Pipeline design")
        st.markdown(
            """
            1. Download target ticker data and `SPY` market context
            2. Engineer technical and relative-performance features
            3. Train models with expanding-window walk-forward validation
            4. Convert out-of-sample probabilities into buy signals
            5. Backtest against passive and naive baseline strategies
            6. Export metrics, charts, and serialized model artifacts
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with arch_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Codebase layout")
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
        st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="footer-note">Live demo built with Streamlit. Full source code, research notes, and project documentation are available in the GitHub repository.</div>',
    unsafe_allow_html=True,
)
