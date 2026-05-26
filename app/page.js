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

const defaultCrossValidation = {
  scheme: "walk_forward",
  folds: [
    {
      fold: 1,
      rows: 173,
      start_date: "2022-03-23 00:00:00",
      end_date: "2022-11-28 00:00:00",
      accuracy: 0.5549132947976878,
      precision: 0.5512820512820513,
      recall: 0.5058823529411764,
      f1: 0.5276073619631901,
      roc_auc: 0.5530748663101605
    },
    {
      fold: 2,
      rows: 173,
      start_date: "2022-11-29 00:00:00",
      end_date: "2023-08-08 00:00:00",
      accuracy: 0.48554913294797686,
      precision: 0.5113636363636364,
      recall: 0.4945054945054945,
      f1: 0.5027932960893855,
      roc_auc: 0.49597963012597157
    },
    {
      fold: 3,
      rows: 173,
      start_date: "2023-08-09 00:00:00",
      end_date: "2024-04-16 00:00:00",
      accuracy: 0.4682080924855491,
      precision: 0.4406779661016949,
      recall: 0.3058823529411765,
      f1: 0.3611111111111111,
      roc_auc: 0.46510695187165774
    },
    {
      fold: 4,
      rows: 173,
      start_date: "2024-04-17 00:00:00",
      end_date: "2024-12-20 00:00:00",
      accuracy: 0.4161849710982659,
      precision: 0.6666666666666666,
      recall: 0.18018018018018017,
      f1: 0.28368794326241137,
      roc_auc: 0.5010171461784365
    }
  ],
  summary: {
    mean_accuracy: 0.48121387283236994,
    mean_precision: 0.5424975801035123,
    mean_recall: 0.3716125951420069,
    mean_f1: 0.41879992810652455,
    mean_roc_auc: 0.5037946486215565,
    fold_count: 4
  }
};

const defaultBatchLeaderboard = {
  aggregate: {
    model_name: "xgboost",
    tickers: ["AAPL", "MSFT", "NVDA"],
    start_date: "2018-01-01",
    end_date: "2026-05-25",
    experiment_count: 3,
    failure_count: 0,
    mean_strategy_return: 2.7460104831809104,
    median_strategy_return: 0.2003229924190757,
    mean_sharpe_ratio: 0.9270887090542324,
    mean_cv_accuracy: 0.5012019230769231,
    mean_cv_roc_auc: 0.5072337687228755
  },
  experiments: [
    {
      ticker: "AAPL",
      model_name: "xgboost",
      classification_accuracy: 0.46634615384615385,
      classification_precision: 0.5115384615384615,
      classification_recall: 0.4117647058823529,
      classification_f1: 0.45626171875,
      classification_roc_auc: 0.47836538461538464,
      mean_cv_accuracy: 0.47716346153846156,
      mean_cv_precision: 0.5183102737325996,
      mean_cv_recall: 0.3699486963421979,
      mean_cv_f1: 0.4295328994349372,
      mean_cv_roc_auc: 0.4787961092656249,
      best_threshold: 0.5,
      strategy_return: 0.06335364713279512,
      sharpe_ratio: 0.1335177431672168,
      max_drawdown: -0.2195388144974485,
      trade_count: 319,
      buy_and_hold_return: 1.1106937499087626
    },
    {
      ticker: "MSFT",
      model_name: "xgboost",
      classification_accuracy: 0.5144230769230769,
      classification_precision: 0.51440329218107,
      classification_recall: 0.5631067961165048,
      classification_f1: 0.5376543209876543,
      classification_roc_auc: 0.5216828478964401,
      mean_cv_accuracy: 0.5132211538461539,
      mean_cv_precision: 0.5476877651465799,
      mean_cv_recall: 0.49139793041421415,
      mean_cv_f1: 0.516010955706382,
      mean_cv_roc_auc: 0.5136984027629816,
      best_threshold: 0.65,
      strategy_return: 0.2003229924190757,
      sharpe_ratio: 0.41716966465571187,
      max_drawdown: -0.1387822819765273,
      trade_count: 183,
      buy_and_hold_return: 0.745980729441563
    },
    {
      ticker: "NVDA",
      model_name: "xgboost",
      classification_accuracy: 0.5721153846153846,
      classification_precision: 0.6041666666666666,
      classification_recall: 0.5272727272727272,
      classification_f1: 0.5631067961165048,
      classification_roc_auc: 0.5494434137291281,
      mean_cv_accuracy: 0.5132211538461539,
      mean_cv_precision: 0.5949317226890756,
      mean_cv_recall: 0.3659504746461268,
      mean_cv_f1: 0.4469487944072876,
      mean_cv_roc_auc: 0.5292067941400201,
      best_threshold: 0.5,
      strategy_return: 7.97435480999086,
      sharpe_ratio: 2.2305787193397686,
      max_drawdown: -0.16528042969633527,
      trade_count: 280,
      buy_and_hold_return: 10.583389680263446
    }
  ],
  failures: []
};

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function ratio(value) {
  return value.toFixed(2);
}

function median(values) {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
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
  const [crossValidation, setCrossValidation] = useState(defaultCrossValidation);
  const [batchLeaderboard, setBatchLeaderboard] = useState(defaultBatchLeaderboard);
  const [activeVisual, setActiveVisual] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryRes, benchmarksRes, signalsRes, crossValidationRes, batchLeaderboardRes] = await Promise.all([
          fetch("/demo/aapl_xgboost_summary.json"),
          fetch("/demo/aapl_xgboost_benchmarks.json"),
          fetch("/demo/aapl_latest_signals.json"),
          fetch("/demo/aapl_xgboost_cross_validation.json"),
          fetch("/demo/batch_xgboost_leaderboard.json")
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
        if (crossValidationRes.ok) {
          setCrossValidation(await crossValidationRes.json());
        }
        if (batchLeaderboardRes.ok) {
          setBatchLeaderboard(await batchLeaderboardRes.json());
        }
      } catch (_error) {
        // Keep baked-in demo fallbacks if the API is unavailable.
      }
    }

    loadData();
  }, []);

  const rows = useMemo(() => benchmarkRows(summary, benchmarks), [summary, benchmarks]);
  const foldRows = useMemo(
    () =>
      crossValidation.folds.map((fold) => ({
        fold: fold.fold,
        window: `${fold.start_date.slice(0, 10)} to ${fold.end_date.slice(0, 10)}`,
        accuracy: pct(fold.accuracy),
        precision: pct(fold.precision),
        recall: pct(fold.recall),
        f1: pct(fold.f1),
        rocAuc: ratio(fold.roc_auc)
      })),
    [crossValidation]
  );
  const batchRows = useMemo(
    () =>
      batchLeaderboard.experiments.map((experiment) => ({
        ticker: experiment.ticker,
        bestThreshold: experiment.best_threshold.toFixed(2),
        strategyReturn: pct(experiment.strategy_return),
        buyAndHoldReturn: pct(experiment.buy_and_hold_return),
        sharpeRatio: ratio(experiment.sharpe_ratio),
        cvAccuracy: pct(experiment.mean_cv_accuracy),
        cvRocAuc: ratio(experiment.mean_cv_roc_auc),
        tradeCount: experiment.trade_count
      })),
    [batchLeaderboard]
  );
  const medianStrategyReturn = useMemo(() => {
    const fromAggregate = batchLeaderboard.aggregate.median_strategy_return;
    if (typeof fromAggregate === "number") {
      return fromAggregate;
    }
    return median(batchLeaderboard.experiments.map((experiment) => experiment.strategy_return));
  }, [batchLeaderboard]);

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
        <MetricCard label="Walk-Forward Accuracy" value={pct(crossValidation.summary.mean_accuracy)} />
      </section>

      <section className="section-stack">
        <div className="section-header">
          <h2>Predictive Quality</h2>
          <p>Walk-forward classification metrics showing how well the model predicts next-day direction out of sample.</p>
        </div>
        <div className="metric-grid validation-grid">
          <MetricCard label="Mean Accuracy" value={pct(crossValidation.summary.mean_accuracy)} />
          <MetricCard label="Mean Precision" value={pct(crossValidation.summary.mean_precision)} />
          <MetricCard label="Mean Recall" value={pct(crossValidation.summary.mean_recall)} />
          <MetricCard label="Mean ROC-AUC" value={ratio(crossValidation.summary.mean_roc_auc)} />
        </div>
        <div className="table-card fold-table-card">
          <table>
            <thead>
              <tr>
                <th>Fold</th>
                <th>Validation Window</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {foldRows.map((row) => (
                <tr key={row.fold}>
                  <td>{row.fold}</td>
                  <td>{row.window}</td>
                  <td>{row.accuracy}</td>
                  <td>{row.precision}</td>
                  <td>{row.recall}</td>
                  <td>{row.f1}</td>
                  <td>{row.rocAuc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
          <h2>Cross-Ticker Study</h2>
          <p>
            Current public XGBoost basket across {batchLeaderboard.aggregate.experiment_count} completed names:
            {" "}
            {batchLeaderboard.aggregate.tickers.join(", ")}.
          </p>
        </div>
        <div className="metric-grid validation-grid">
          <MetricCard label="Completed Tickers" value={String(batchLeaderboard.aggregate.experiment_count)} />
          <MetricCard label="Median Strategy Return" value={pct(medianStrategyReturn)} tone={cls(medianStrategyReturn)} />
          <MetricCard label="Mean CV Accuracy" value={pct(batchLeaderboard.aggregate.mean_cv_accuracy)} />
          <MetricCard label="Mean CV ROC-AUC" value={ratio(batchLeaderboard.aggregate.mean_cv_roc_auc)} />
        </div>
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Best Threshold</th>
                <th>Strategy Return</th>
                <th>Buy & Hold</th>
                <th>Sharpe Ratio</th>
                <th>CV Accuracy</th>
                <th>CV ROC-AUC</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              {batchRows.map((row) => (
                <tr key={row.ticker}>
                  <td>{row.ticker}</td>
                  <td>{row.bestThreshold}</td>
                  <td>{row.strategyReturn}</td>
                  <td>{row.buyAndHoldReturn}</td>
                  <td>{row.sharpeRatio}</td>
                  <td>{row.cvAccuracy}</td>
                  <td>{row.cvRocAuc}</td>
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
          <p>Latest public AAPL XGBoost research outputs from the Python pipeline.</p>
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
            src="/images/aapl_xgboost_probabilities.png"
            alt="AAPL XGBoost Probability Distribution"
            title="XGBoost Probability Distribution"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_xgboost_probabilities.png",
                alt: "AAPL XGBoost Probability Distribution",
                title: "XGBoost Probability Distribution"
              })
            }
          />
        </div>
        <div className="visual-grid visual-grid-square">
          <VisualCard
            src="/images/aapl_xgboost_confusion_matrix.png"
            alt="XGBoost Confusion Matrix"
            title="XGBoost Confusion Matrix"
            onOpen={() =>
              setActiveVisual({
                src: "/images/aapl_xgboost_confusion_matrix.png",
                alt: "XGBoost Confusion Matrix",
                title: "XGBoost Confusion Matrix"
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
