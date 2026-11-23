"""Score public financial headlines locally, keeping the original observation clock."""
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests

from .news import append_versions, utc

REVISION = "8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20"
SCORER = "Xenova/finbert@" + REVISION + ":title-quotes-v1"
MODEL_FOLDER = Path("data/news_v9/finbert") / REVISION
FILES = {
    "model_quantized.onnx": "56cd19c7068ca5670c326816845bb7d7edda508b68a228697a1460d0e821ce76",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "3629c9707884176d61af8452d8779ee167169db85d81e2183afe939a88c483b8",
}


def prepare_model(folder=MODEL_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = folder / name
        if path.exists() and sha256(path.read_bytes()).hexdigest() == expected:
            continue
        remote = ("onnx/" if name.endswith(".onnx") else "") + name
        url = f"https://huggingface.co/Xenova/finbert/resolve/{REVISION}/{remote}"
        with NamedTemporaryFile(dir=folder, delete=False) as handle:
            temp = Path(handle.name)
        try:
            digest = sha256()
            with requests.get(url, timeout=(8, 60), stream=True) as response, temp.open("wb") as out:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Unexpected digest for {name}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return folder


class SentimentModel:
    def __init__(self):
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer
        folder = prepare_model()
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        opts = SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = InferenceSession(str(folder / "model_quantized.onnx"), opts, providers=["CPUExecutionProvider"])
        self.labels = json.loads((folder / "config.json").read_text())["id2label"]

    def score(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        inputs = {name: np.array([getattr(e, attr) for e in encoded], dtype=np.int64)
                  for name, attr in [("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                     ("token_type_ids", "type_ids")]}
        logits = self.session.run(None, inputs)[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return [{self.labels[str(i)]: float(p) for i, p in enumerate(row)}
                for row in exp / exp.sum(axis=1, keepdims=True)]


def ledger_rows(path):
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def score_pending(folder):
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "sentiment.jsonl"
    with (folder / ".sentiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = {r["article_version"] for r in ledger_rows(output) if r["scorer"] == SCORER}
        with (folder / ".ledger.lock").open("a") as source_lock:
            fcntl.flock(source_lock, fcntl.LOCK_SH)
            pending = [r for r in ledger_rows(folder / "articles.jsonl") if r["version_id"] not in done]
        if not pending:
            return 0
        model = SentimentModel()
        added = 0
        for start in range(0, len(pending), 16):
            batch = pending[start:start + 16]
            scores = model.score([r["title"] for r in batch])
            quotes = [q for r in batch for q in r.get("company_quotes", [])]
            quote_scores = iter([s for offset in range(0, len(quotes), 16)
                                 for s in model.score(quotes[offset:offset+16])])
            now = datetime.now(timezone.utc).isoformat()
            rows = []
            for article, probabilities in zip(batch, scores):
                key = article["version_id"]
                rows.append({"version_id": sha256((SCORER + key).encode()).hexdigest(),
                             "article_version": key, "scorer": SCORER, "scored_at": now,
                             "probabilities": probabilities,
                             "quote_probabilities": [next(quote_scores) for _ in article.get("company_quotes", [])]})
            added += append_versions(output, rows)
        return added


def sentiment_at(cutoff, folder, activity):
    """Score latest eligible versions for URLs first observed in the supplied window."""
    from datetime import timedelta
    end = utc(cutoff)
    return sentiment_window(folder, end - timedelta(hours=1), end,
                            complete=activity["recent_batches"] == activity["expected_batches_per_hour"])


def sentiment_window(folder, start, end, *, complete):
    """Use news versions and model scores observable by the end of this window."""
    from .press_news import HEADLINES
    fields = {"eligible_urls": None, "headline_scored": None, "headline_balance": None,
              "article_tone_scored": None, "article_tone_mean": None,
              "articles_with_scored_quotes": None, "quote_balance": None}
    result = {t: fields.copy() for t in HEADLINES}
    if not complete:
        return result
    scores = {r["article_version"]: r for r in ledger_rows(folder / "sentiment.jsonl")
              if r["scorer"] == SCORER and utc(r["scored_at"]) <= end}
    versions = {}
    for row in ledger_rows(folder / "articles.jsonl"):
        if utc(row["available_at"]) <= end:
            versions.setdefault((row["ticker"], row["url"]), []).append(row)
    eligible = {t: [] for t in HEADLINES}
    for (ticker, _), rows in versions.items():
        rows.sort(key=lambda r: utc(r["available_at"]))
        if start <= utc(rows[0]["gdelt_batch_at"]) < end:
            row = rows[-1]
            # Mixed-company headlines cannot safely assign one story's tone to both sides.
            if sum(bool(pattern.search(row["title"])) for pattern in HEADLINES.values()) == 1:
                eligible[ticker].append(row)
    for ticker, rows in eligible.items():
        scored = [scores[r["version_id"]] for r in rows if r["version_id"] in scores]
        prob = [s["probabilities"] for s in scored]
        quote_means = [np.mean([q["positive"] - q["negative"] for q in s["quote_probabilities"]])
                       for s in scored if s["quote_probabilities"]]
        tones = [r["article_tone"] for r in rows if r.get("article_tone") is not None]
        result[ticker] = {"eligible_urls": len(rows), "headline_scored": len(prob),
                          "headline_balance": float(np.mean([s["positive"] - s["negative"] for s in prob])) if prob else None,
                          "article_tone_scored": len(tones), "article_tone_mean": float(np.mean(tones)) if tones else None,
                          "articles_with_scored_quotes": len(quote_means),
                          "quote_balance": float(np.mean(quote_means)) if quote_means else None}
    return result
