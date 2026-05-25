"use client";

import { useEffect, useMemo, useState } from "react";

const defaultSummary = {
  total_return: -0.3686639960773508,
  max_drawdown: -0.43869492254445686,
  sharpe_ratio: -1.0196621431557589,
  win_rate: 0.45348837209302323,
  trade_count: 172,
  benchmark_return: 0.4816990012356057
};

const defaultBenchmarks = {
  buy_and_hold: {
    total_return: 0.4816990012356057,
    max_drawdown: -0.30140818086268584,
    sharpe_ratio: 0.6280743522557665,
    win_rate: 0.5373563218390804,
    trade_count: 696
  },
  always_long: {
    total_return: 0.04628895071763006,
    max_drawdown: -0.3661348113110008,
    sharpe_ratio: 0.15762110072656865,
    win_rate: 0.5244252873563219,
    trade_count: 696
  },
  momentum: {
    total_return: -0.1160872622654644,
    max_drawdown: -0.350941546506064,
    sharpe_ratio: -0.20443908929983176,
    win_rate: 0.28304597701149425,
    trade_count: 374
  }
};

const defaultSignals = [
  {
    ticker: "AAPL",
    model_name: "logistic_regression",
    as_of_date: "2026-05-22 00:00:00",
    prediction: 1,
    prob_up: 0.5409567919713436,
    signal: 0
  },
  {
    ticker: "AAPL",
    model_name: "random_forest",
    as_of_date: "2026-05-22 00:00:00",
    prediction: 1,
    prob_up: 0.5578161083352521,
    signal: 1
  },
  {
    ticker: "AAPL",
    model_name: "xgboost",
    as_of_date: "2026-05-22 00:00:00",
    prediction: 1,
    prob_up: 0.6176891922950745,
    signal: 1
  },
  {
    ticker: "AAPL",
    model_name: "lstm",
    as_of_date: "2026-05-22 00:00:00",
    prediction: 1,
    prob_up: 0.639611542224884,
    signal: 1
  }
];

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function ratio(value) {
  return value.toFixed(2);
}

function cls(value) {
  return value >= 0 ? "positive" : "negative";
}

const benchmarkRows = (summary, benchmarks) => [
  {
    strategy: "Model Strategy",
    totalReturn: pct(summary.total_return),
    maxDrawdown: pct(summary.max_drawdown),
    sharpeRatio: ratio(summary.sharpe_ratio),
    winRate: pct(summary.win_rate),
    tradeCount: summary.trade_count
  },
  {
    strategy: "Buy & Hold",
    totalReturn: pct(benchmarks.buy_and_hold.total_return),
    maxDrawdown: pct(benchmarks.buy_and_hold.max_drawdown),
    sharpeRatio: ratio(benchmarks.buy_and_hold.sharpe_ratio),
    winRate: pct(benchmarks.buy_and_hold.win_rate),
    tradeCount: benchmarks.buy_and_hold.trade_count
  },
  {
    strategy: "Always Long",
    totalReturn: pct(benchmarks.always_long.total_return),
    maxDrawdown: pct(benchmarks.always_long.max_drawdown),
    sharpeRatio: ratio(benchmarks.always_long.sharpe_ratio),
    winRate: pct(benchmarks.always_long.win_rate),
    tradeCount: benchmarks.always_long.trade_count
  },
  {
    strategy: "Momentum",
    totalReturn: pct(benchmarks.momentum.total_return),
    maxDrawdown: pct(benchmarks.momentum.max_drawdown),
    sharpeRatio: ratio(benchmarks.momentum.sharpe_ratio),
    winRate: pct(benchmarks.momentum.win_rate),
    tradeCount: benchmarks.momentum.trade_count
  }
];

export default function HomePage() {
  const [summary, setSummary] = useState(defaultSummary);
  const [benchmarks, setBenchmarks] = useState(defaultBenchmarks);
  const [signals, setSignals] = useState(defaultSignals);
  const [activeVisual, setActiveVisual] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryRes, benchmarksRes, signalsRes] = await Promise.all([
          fetch("/api/demo/summary"),
          fetch("/api/demo/benchmarks"),
          fetch("/api/demo/live-signals")
        ]);

        if (summaryRes.ok) {
          setSummary(await summaryRes.json());
        }
        if (benchmarksRes.ok) {
          setBenchmarks(await benchmarksRes.json());
        }
        if (signalsRes.ok) {
          setSignals(await signalsRes.json());
        }
      } catch (_error) {
        // Keep baked-in demo fallbacks if the API is unavailable.
      }
    }

    loadData();
  }, []);

  const rows = useMemo(() => benchmarkRows(summary, benchmarks), [summary, benchmarks]);

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div className="eyebrow">Quantitative Machine Learning Research Platform</div>
        <h1>QuantLab AI</h1>
        <p className="hero-copy">
          A production-style quant research system for studying whether stock-direction models generate
          signals that survive walk-forward validation, benchmark comparison, and realistic backtesting.
        </p>
        <div className="hero-callout">
          Classification signal and tradable edge are not the same thing. QuantLab AI is built to measure
          that gap honestly.
        </div>
        <div className="tag-row">
          <span>Walk-Forward Validation</span>
          <span>Benchmark-Aware Backtesting</span>
          <span>Market Context Features</span>
          <span>scikit-learn + XGBoost + PyTorch</span>
        </div>
      </section>

      <section className="metric-grid">
        <MetricCard label="Ticker" value="AAPL" />
        <MetricCard label="Lead Model" value="XGBoost" />
        <MetricCard label="Strategy Return" value={pct(summary.total_return)} tone={cls(summary.total_return)} />
        <MetricCard label="Buy & Hold Return" value={pct(summary.benchmark_return)} tone="positive" />
      </section>

      <section className="content-grid">
        <article className="card feature-card">
          <h2>Research Overview</h2>
          <p>
            QuantLab AI combines technical indicators, market-relative features, and multiple model families
            to evaluate whether predictive stock signals translate into robust trading performance.
          </p>
          <ul>
            <li>Historical equity data plus SPY market context</li>
            <li>Technical signals, relative strength, beta, and correlation features</li>
            <li>Logistic Regression, Random Forest, XGBoost, and LSTM models</li>
            <li>Expanding-window walk-forward validation</li>
            <li>Backtesting versus buy-and-hold, always-long, and momentum baselines</li>
          </ul>
        </article>

        <article className="card snapshot-card">
          <h2>Benchmark Snapshot</h2>
          <div className="mini-grid">
            <MiniStat label="Strategy Return" value={pct(summary.total_return)} tone={cls(summary.total_return)} />
            <MiniStat label="Buy & Hold" value={pct(summary.benchmark_return)} tone="positive" />
            <MiniStat label="Max Drawdown" value={pct(summary.max_drawdown)} tone="negative" />
            <MiniStat label="Trade Count" value={String(summary.trade_count)} />
          </div>
          <p className="snapshot-note">
            The current public XGBoost baseline outperforms naive momentum but still fails to beat passive
            exposure, which is exactly the kind of result a serious quant workflow should reveal.
          </p>
        </article>
      </section>

      <section className="section-stack">
        <div className="section-header">
          <h2>Benchmark Comparison</h2>
          <p>Walk-forward strategy performance versus baseline rules.</p>
        </div>
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Max Drawdown</th>
                <th>Sharpe Ratio</th>
                <th>Win Rate</th>
                <th>Trade Count</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.strategy}>
                  <td>{row.strategy}</td>
                  <td>{row.totalReturn}</td>
                  <td>{row.maxDrawdown}</td>
                  <td>{row.sharpeRatio}</td>
                  <td>{row.winRate}</td>
                  <td>{row.tradeCount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-stack">
        <div className="section-header">
          <h2>Latest Live Signals</h2>
          <p>Most recent model outputs from the live signal tracking workflow.</p>
        </div>
        <div className="signal-grid">
          {signals.map((signal) => (
            <article key={signal.model_name} className="signal-card">
              <div className="signal-topline">
                <span className="signal-model">{signal.model_name.replaceAll("_", " ")}</span>
                <span className={`signal-pill ${signal.signal === 1 ? "signal-on" : "signal-off"}`}>
                  {signal.signal === 1 ? "Trade Signal" : "Watchlist"}
                </span>
              </div>
              <div className="signal-metric">{pct(signal.prob_up)}</div>
              <div className="signal-label">Probability of upward movement</div>
              <div className="signal-footer">
                <span>Prediction: {signal.prediction === 1 ? "Up" : "Down"}</span>
                <span>{signal.as_of_date.slice(0, 10)}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section-stack">
        <div className="section-header">
          <h2>Visual Analytics</h2>
          <p>Sample research outputs from the Python pipeline.</p>
        </div>
        <div className="visual-grid visual-grid-wide">
          <VisualCard
            src="/images/aapl_xgboost_equity_curve.png"
            alt="AAPL XGBoost Equity Curve"
            title="XGBoost Strategy vs Benchmarks"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_xgboost_equity_curve.png",
                alt: "AAPL XGBoost Equity Curve",
                title: "XGBoost Strategy vs Benchmarks"
              })
            }
          />
          <VisualCard
            src="/images/aapl_lstm_equity_curve.png"
            alt="AAPL LSTM Equity Curve"
            title="LSTM Strategy vs Benchmarks"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_lstm_equity_curve.png",
                alt: "AAPL LSTM Equity Curve",
                title: "LSTM Strategy vs Benchmarks"
              })
            }
          />
        </div>
        <div className="visual-grid visual-grid-square">
          <VisualCard
            src="/images/aapl_random_forest_confusion_matrix.png"
            alt="Random Forest Confusion Matrix"
            title="Random Forest Confusion Matrix"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_random_forest_confusion_matrix.png",
                alt: "Random Forest Confusion Matrix",
                title: "Random Forest Confusion Matrix"
              })
            }
          />
          <VisualCard
            src="/images/aapl_candlestick.png"
            alt="AAPL Candlestick Chart"
            title="AAPL Candlestick Chart"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_candlestick.png",
                alt: "AAPL Candlestick Chart",
                title: "AAPL Candlestick Chart"
              })
            }
          />
        </div>
      </section>

      <section className="section-stack architecture-stack">
        <div className="section-header">
          <h2>System Design</h2>
          <p>How the research pipeline is organized.</p>
        </div>
        <div className="architecture-grid">
          <article className="card">
            <h3>Pipeline</h3>
            <ol>
              <li>Download ticker data and SPY market context</li>
              <li>Engineer technical and market-relative features</li>
              <li>Train models with expanding-window validation</li>
              <li>Generate out-of-sample probabilities and signals</li>
              <li>Backtest against baseline strategies</li>
              <li>Export metrics, charts, and artifacts</li>
            </ol>
          </article>
          <article className="card">
            <h3>Deployment Shape</h3>
            <pre>{`Next.js frontend  ->  Vercel
Python API        ->  Vercel Functions
Research pipeline ->  Local / batch workflow
Artifacts         ->  Committed demo assets + JSON`}</pre>
          </article>
        </div>
      </section>

      {activeVisual ? (
        <div className="lightbox-backdrop" onClick={() => setActiveVisual(null)} role="presentation">
          <div className="lightbox-panel" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <button className="lightbox-close" onClick={() => setActiveVisual(null)} aria-label="Close image">
              ×
            </button>
            <div className="lightbox-image-wrap">
              <img src={activeVisual.src} alt={activeVisual.alt} />
            </div>
            <h3>{activeVisual.title}</h3>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function MetricCard({ label, value, tone }) {
  return (
    <article className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${tone ?? ""}`}>{value}</div>
    </article>
  );
}

function MiniStat({ label, value, tone }) {
  return (
    <div className="mini-stat">
      <div className="mini-label">{label}</div>
      <div className={`mini-value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

function VisualCard({ src, alt, title, onOpen }) {
  return (
    <article className="visual-card">
      <div className="visual-frame">
        <button className="visual-button" onClick={onOpen} aria-label={`Open ${title}`}>
          <img src={src} alt={alt} />
        </button>
      </div>
      <h3>{title}</h3>
      <button className="visual-expand" onClick={onOpen}>
        Expand Chart
      </button>
    </article>
  );
}
