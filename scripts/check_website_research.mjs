import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const snapshot = JSON.parse(readFileSync(resolve(root, "public/research/results.json"), "utf8"));
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
assert.equal(snapshot.schema_version, 2);
assert.equal(snapshot.status, "research_snapshot");
assert.equal(snapshot.studies.length, 3);
let assetsChecked = 0;
for (const study of snapshot.studies) {
  assert.equal(new Set(study.tickers).size, study.tickers.length);
  assert.match(study.source_sha256, /^[a-f0-9]{64}$/);
  assert.ok(study.start < study.end);
  assert.equal(study.evidence_type, study.id === "confirmation" ? "retrospective_confirmation" : "historical_development");
  for (const model of ["xgboost", "logistic_regression"]) {
    const { assets, aggregates } = study.models[model];
    assert.equal(assets.length, study.tickers.length * aggregates.length);
    for (const row of [...assets, ...aggregates]) {
      assert.ok(Number.isInteger(row.rows) && row.rows > 0);
      assert.ok(row.accuracy >= 0 && row.accuracy <= 1);
      assert.ok(row.always_up_accuracy >= 0 && row.always_up_accuracy <= 1);
      assert.equal(row.correct_down + row.incorrect_down, row.down_calls);
      assert.equal(row.correct_down - row.incorrect_down, row.extra_correct_vs_always_up);
      near(row.accuracy_gain, row.extra_correct_vs_always_up / row.rows);
      near(row.accuracy - row.always_up_accuracy, row.accuracy_gain);
      near(row.always_down_accuracy + row.always_up_accuracy, 1);
      near(row.accuracy - row.always_down_accuracy, row.accuracy_gain_vs_always_down);
      assert.equal(row.interval.length, 2);
      assert.ok(row.interval.every(Number.isFinite) && row.interval[0] <= row.interval[1]);
    }
    for (const aggregate of aggregates) {
      const members = assets.filter((row) => row.scope === aggregate.scope);
      assert.deepEqual([...members.map((row) => row.ticker)].sort(), [...study.tickers].sort());
      assert.ok(members.every((row) => row.rows === study.sessions_per_asset));
      for (const key of ["rows", "down_calls", "correct_down", "incorrect_down", "extra_correct_vs_always_up"]) {
        assert.equal(members.reduce((sum, row) => sum + row[key], 0), aggregate[key]);
      }
      near(members.reduce((sum, row) => sum + row.accuracy * row.rows, 0) / aggregate.rows, aggregate.accuracy);
      near(members.reduce((sum, row) => sum + row.always_up_accuracy * row.rows, 0) / aggregate.rows, aggregate.always_up_accuracy);
    }
    assetsChecked += assets.length;
  }
}
const confirmation = snapshot.studies.find((study) => study.id === "confirmation");
assert.equal(confirmation.start, "2026-05-26");
assert.equal(confirmation.end, "2026-09-18");
assert.equal(confirmation.sessions_per_asset, 81);
const selected = confirmation.models.logistic_regression.aggregates.find((row) => row.scope === "validation_selected");
near(selected.accuracy, 126 / 243);
near(selected.always_up_accuracy, 116 / 243);
near(selected.always_down_accuracy, 127 / 243);
assert.ok(selected.accuracy_gain > 0 && selected.accuracy_gain_vs_always_down < 0);
assert.ok(snapshot.studies.filter((study) => study.id !== "confirmation").every((study) => study.end < confirmation.start));
for (const report of Object.values(snapshot.reports)) {
  assert.match(report, /^\/research\/reports\/[a-z]+\.md$/);
  assert.ok(existsSync(resolve(root, `public${report}`)), `Missing report: ${report}`);
}
console.log(`Verified ${assetsChecked} asset/model/scope rows, all aggregate accounting, and ${Object.keys(snapshot.reports).length} downloadable reports.`);
