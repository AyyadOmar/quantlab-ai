"use client";

import { useState } from "react";
import research from "../public/research/results.json";

const repository = "https://github.com/AyyadOmar/quantlab-ai";
const modelNames = { xgboost: "XGBoost", logistic_regression: "Logistic regression" };
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const points = (value) => `${value > 0.000001 ? "+" : value < -0.000001 ? "−" : ""}${Math.abs(value * 100).toFixed(2)}`;
const number = (value) => value.toLocaleString("en-US");
const dateLabel = (value) => new Date(`${value}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

const experiments = [
  { id: "baseline", number: "01", title: "Start with a fair test", tag: "Method corrected", text: "Predict the next session’s open-to-close direction. Separate training, validation and testing, with realistic trading costs." },
  { id: "features", number: "02", title: "Simplify the inputs", tag: "Probability quality improved", text: "Relative price features and smaller, regularized models improved on the original models. An advantage over always-up remained unproven." },
  { id: "context", number: "03", title: "Add market context", tag: "No reliable advantage", text: "Test sector performance, volatility, interest rates and earnings-related filing recency. The additional inputs did not establish a dependable edge." },
  { id: "earnings", number: "04", title: "Test earnings information", tag: "Exploratory evidence", text: "114 reported earnings records and 34 NVIDIA advance-call notices. Current estimate snapshots may contain revisions; calendar coverage is incomplete." },
  { id: "direction", number: "05", title: "Make better down calls", tag: "Below the benchmark", text: "Choose the classification cutoff on earlier validation data, with always-up as a fallback. Incorrect down calls outweighed correct ones." },
  { id: "shared", number: "06", title: "Learn across companies", tag: "Below the benchmark", text: "Train on Apple, Microsoft and NVIDIA together. Compare with separate models using identical features, settings and evaluation dates." },
  { id: "recent", number: "07", title: "Use more recent history", tag: "Limited confirmation", text: "Shorter training histories did not help overall in development. Newer-period logistic results beat always-up, but trailed always-down; all gains over always-up came from NVIDIA." },
  { id: "intraday", number: "08", title: "Match inputs to the session", tag: "No improvement", text: "Add recent open-to-close returns and intraday up frequency. Accuracy and probability quality declined for both model families. The default inputs stay unchanged." },
  { id: "news", number: "09", title: "Collect company news", tag: "Data collection only", text: "Timestamp public company announcements before testing event information. Microsoft’s legacy feed is stale. No news model has been trained or accuracy benefit measured." },
];

export default function HomePage() {
  const [studyId, setStudyId] = useState("confirmation");
  const [model, setModel] = useState("logistic_regression");
  const study = research.studies.find((item) => item.id === studyId);
  const selection = study.models[model];
  const first = selection.aggregates[0];
  const shared = studyId === "shared";
  const confirmation = studyId === "confirmation";

  return (
    <>
      <a className="skip-link" href="#results">Skip to research results</a>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="QuantLab AI home"><img src="/favicon.png" width="34" height="34" alt="" /><span>QuantLab <b>AI</b></span></a>
        <nav aria-label="Main navigation"><a href="#results">Results</a><a href="#method">Method</a><a href="#research">Research log</a></nav>
        <a className="repo-link" href={repository} target="_blank" rel="noreferrer">View source <span aria-hidden="true">↗</span></a>
      </header>
      <main id="top" className="page-shell">
        <section className="hero" aria-labelledby="hero-title">
          <div>
            <p className="eyebrow"><span className="small-rule" /> Quantitative machine learning research</p>
            <h1 id="hero-title">Can a model beat<br /><em>“always up”?</em></h1>
            <p className="hero-copy">A research notebook on stock prediction. Every model is measured against always-up and always-down, on the same days, with the full result in view.</p>
            <div className="hero-actions"><a className="button primary" href="#results">Explore the evidence <span aria-hidden="true">↓</span></a><a className="text-link" href="/research/reports/protocol.md" download>Read the protocol <span aria-hidden="true">↗</span></a></div>
          </div>
          <aside className="finding-card" aria-label="Current research finding">
            <div className="finding-top"><span className="eyebrow">Current finding</span><span className="status-dot" aria-hidden="true" /></div>
            <h2>No reliable<br />advantage. <em>Yet.</em></h2>
            <p>The newer logistic model beat always-up but fell short of always-down. Its improvement came entirely from NVIDIA.</p>
            <div className="finding-footer"><span>Research snapshot</span><span>Outcomes through Sep 18, 2026</span></div>
          </aside>
        </section>
        <div className="research-strip"><span><i aria-hidden="true" /> Public data · Reproducible studies</span><span>Next-session open → close</span><span>Prospective evidence still required</span></div>

        <section id="results" className="section-block" aria-labelledby="results-title">
          <div className="section-heading"><div><p className="eyebrow">01 / The evidence</p><h2 id="results-title">Same days. Both baselines.</h2></div><p>Always-up predicts an up session; always-down predicts a down or flat session. Neither uses market information.</p></div>
          <div className="results-panel">
            <div className="result-controls">
              <div className="study-switch" role="group" aria-label="Choose research study">
                {research.studies.map((item) => <button key={item.id} type="button" aria-pressed={studyId === item.id} onClick={() => setStudyId(item.id)}>{item.title}<span>{item.id === "confirmation" ? "Retrospective · 2026" : "Historical development"}</span></button>)}
              </div>
              <label className="model-control">Model<select value={model} onChange={(event) => setModel(event.target.value)}>{Object.entries(modelNames).map(([key, name]) => <option key={key} value={key}>{name}</option>)}</select></label>
            </div>
            <div className="result-body" aria-live="polite" aria-atomic="true">
              <div className="study-context"><span>{study.tickers.join(" · ")}</span><span>{dateLabel(study.start)} — {dateLabel(study.end)} · {study.sessions_per_asset} sessions per asset · {number(first.rows)} predictions</span></div>
              <div className="evidence-note"><strong>{confirmation ? "Newer-period retrospective confirmation" : "Historical development"}</strong><p>{confirmation ? "Model choices were frozen before newer prices were retrieved, but these predictions were reconstructed after the events. This is not a record of live forecasts. All ten extra correct logistic predictions versus always-up came from NVIDIA; Apple and Microsoft matched always-up." : "These dates have been used in earlier experiments. Results support development and comparison; they are not independent confirmation."}</p></div>
              <div className={`metric-grid ${selection.aggregates.length === 2 ? "four-metrics" : "three-metrics"}`}>
                <Metric label="Always-up baseline" value={percent(first.always_up_accuracy)} note="Predict up every session" benchmark />
                <Metric label="Always-down baseline" value={percent(first.always_down_accuracy)} note="Predict down or flat every session" benchmark />
                {selection.aggregates.map((row) => <Metric key={row.scope} label={row.label} value={percent(row.accuracy)} note={`${points(row.accuracy_gain)} pp vs up · ${points(row.accuracy_gain_vs_always_down)} pp vs down`} />)}
              </div>
              <div className="analysis-grid">
                <div className="chart-section">
                  <div className="subheading"><h3>Accuracy, in perspective</h3><span>0–100% scale</span></div>
                  <ComparisonBar label="Always-up" value={first.always_up_accuracy} benchmark />
                  <ComparisonBar label="Always-down" value={first.always_down_accuracy} benchmark />
                  {selection.aggregates.map((row) => <ComparisonBar key={row.scope} label={row.label} value={row.accuracy} />)}
                  <p className="chart-note">{confirmation ? "The training-history choice uses earlier validation data. Both models are frozen throughout this newer period. A lower always-up score reflects fewer up days, so compare models within the same period." : shared ? "Identical compact features and regularized settings. Only the training data is shared." : "The model is selected using validation probability quality. Its up/down cutoff is then selected using validation accuracy."}</p>
                </div>
                <div className="down-audit">
                  <p className="eyebrow">What changed the score?</p><h3>Every down call counts.</h3>
                  <p>A correct down call fixes one always-up error. An incorrect one creates a new error.</p>
                  {selection.aggregates.map((row) => <div className="audit-row" key={row.scope}><span>{row.label}</span><strong>{row.correct_down} <small>correct</small><span aria-hidden="true"> / </span>{row.incorrect_down} <small>incorrect</small></strong></div>)}
                  <p className="audit-footnote">All test days are included. No low-confidence days are removed.</p>
                </div>
              </div>
              <div className="asset-section"><div className="subheading"><h3>Look beneath the average</h3><span>{modelNames[model]}</span></div>
                <div className="table-scroll" role="region" aria-label="Accuracy by asset, horizontally scrollable" tabIndex={0}>
                  <table><caption className="sr-only">{study.title}: {modelNames[model]} accuracy on the same {study.sessions_per_asset} test sessions per asset</caption><thead><tr><th scope="col">Asset</th><th scope="col">Always-up</th><th scope="col">Always-down</th>{selection.aggregates.map((row) => <th scope="col" key={row.scope}>{row.label}</th>)}<th scope="col">{confirmation ? "Selected-history" : shared ? "Shared-model" : "Selected-rule"} vs up</th></tr></thead>
                    <tbody>{study.tickers.map((ticker) => {
                      const rows = selection.aggregates.map((aggregate) => selection.assets.find((row) => row.ticker === ticker && row.scope === aggregate.scope));
                      const last = rows[rows.length - 1];
                      return <tr key={ticker}><th scope="row">{ticker}</th><td>{percent(rows[0].always_up_accuracy)}</td><td>{percent(rows[0].always_down_accuracy)}</td>{rows.map((row) => <td key={row.scope}>{percent(row.accuracy)}</td>)}<td className={last.accuracy_gain < 0 ? "negative" : "muted"}>{points(last.accuracy_gain)} pp</td></tr>;
                    })}</tbody>
                  </table>
                </div>
              </div>
              <details className="uncertainty"><summary>How certain are these results?<span aria-hidden="true">+</span></summary><div><p>{confirmation ? "Only 81 market sessions are included, and the result is concentrated in one company. These retrospective results have now been inspected and are no longer untouched evidence." : "These historical dates have already been examined in earlier studies."} The intervals describe uncertainty; they do not account for every experiment tried or establish a reliable advantage.</p>{selection.aggregates.map((row) => <p key={row.scope}><strong>{row.label}:</strong> {points(row.interval[0])} to {points(row.interval[1])} percentage points versus always-up (95% descriptive interval).</p>)}<p>We resample 20-session blocks 2,000 times, keeping assets on each date together. Correlated companies are not independent replications. “pp” means percentage points.</p></div></details>
              <div className="result-footer"><span>{confirmation ? "Always-down is shown as a fixed comparison, not a rule chosen for deployment from these outcomes. No default model was promoted." : shared ? "Three-company study. Its baseline differs from the five-asset study." : "Five-asset study. Its baseline differs from the three-company study."}</span><a className="text-link" href={study.report} download>Download this report <span aria-hidden="true">↓</span></a></div>
            </div>
          </div>
        </section>

        <section id="method" className="section-block" aria-labelledby="method-title">
          <div className="section-heading"><div><p className="eyebrow">02 / The method</p><h2 id="method-title">Keep the future out of training.</h2></div><p>The question is specific: will the next session<br />close above its opening price?</p></div>
          <div className="method-grid">
            <article><span className="step-number">01</span><h3>Learn from the past</h3><p>Build price and market features after a completed trading session. Fit the model and preprocessing on earlier training data.</p><span className="method-label">Training window</span></article>
            <article><span className="step-number">02</span><h3>Choose the rule</h3><p>Use a later validation window to select settings and the classification cutoff. Always-up remains an eligible fallback.</p><span className="method-label">Validation window</span></article>
            <article><span className="step-number">03</span><h3>Score later days</h3><p>Freeze those choices before testing on the next window. Leave gaps so earlier label outcomes cannot overlap later windows.</p><span className="method-label">Test window</span></article>
          </div>
          <div className="method-notes"><div><h3>Accuracy is not profit.</h3><p>These latest studies evaluate direction, not a trading strategy. Separate backtests assume 5 basis points in fees and 2 in slippage per side. A higher accuracy does not guarantee higher returns.</p></div><div><h3>Historical evidence has limits.</h3><p>Repeated research can overfit a familiar period. A model needs a frozen specification and confirmation on untouched dates before we claim a dependable advantage.</p></div></div>
        </section>

        <section id="research" className="section-block" aria-labelledby="research-title">
          <div className="section-heading"><div><p className="eyebrow">03 / The research log</p><h2 id="research-title">What we tried. What we learned.</h2></div><a className="text-link" href="/research/results.json" download>Download result data <span aria-hidden="true">↓</span></a></div>
          <div className="experiment-grid">{experiments.map((experiment) => <article className="experiment-card" key={experiment.id}><div className="experiment-top"><span className="experiment-number">{experiment.number}</span><span className="experiment-tag">{experiment.tag}</span></div><h3>{experiment.title}</h3><p>{experiment.text}</p><a href={research.reports[experiment.id]} download>Read study <span className="sr-only">— {experiment.title}</span><span aria-hidden="true">↗</span></a></article>)}</div>
        </section>
        <section className="closing-note"><div><p className="eyebrow">The next standard of proof</p><h2>A better score must survive new data.</h2><p>News collection is experimental. Microsoft’s feed is stale, coverage is incomplete, and no news-based accuracy improvement has been measured. Next comes better data coverage and evaluation of forecasts recorded before their outcomes.</p></div><a className="button secondary" href={repository} target="_blank" rel="noreferrer">Explore the project <span aria-hidden="true">↗</span></a></section>
        <details className="legacy-note"><summary>About the earlier website results</summary><p>The previous demo used an older close-to-close prediction target and different evaluation assumptions. Its returns, charts and dated sample signals are not comparable with the current open-to-close studies. They remain in the <a href={`${repository}/tree/main/docs/demo`} target="_blank" rel="noreferrer">repository archive</a> for reference. This page presents a published research snapshot, not a live signal feed.</p></details>
      </main>
      <footer className="site-footer"><a className="brand" href="#top">QuantLab <b>AI</b></a><p>Independent research. Transparent comparisons.</p><a href={repository} target="_blank" rel="noreferrer">Code & methodology <span aria-hidden="true">↗</span></a></footer>
    </>
  );
}

function Metric({ label, value, note, benchmark = false }) {
  return <div className={`metric ${benchmark ? "benchmark" : ""}`}><p className="metric-label">{label}</p><div className="metric-value">{value}</div><p className="metric-note">{note}</p></div>;
}
function ComparisonBar({ label, value, benchmark = false }) {
  return <div className={`comparison ${benchmark ? "benchmark" : ""}`}><div className="comparison-label"><span>{label}</span><strong>{percent(value)}</strong></div><div className="bar-track" aria-hidden="true"><div className="bar-fill" style={{ width: `${value * 100}%` }} /></div></div>;
}
