"""Score E3: SemanticPass@k against what the user actually did next.

    python score_suggestions.py --run <run_id> [--tau 0.55]

Suggestions are free text, so exact match is meaningless. We embed each
suggestion and the user's real next utterance, and count a hit when cosine
similarity >= tau.

Two things to keep honest:

* **tau is a researcher degree of freedom.** Calibrate it once, write it down,
  and report `mean_max_sim` (threshold-free) alongside every pass@k. Use
  `--calibrate` to see the distribution before choosing.
* The **deployed baseline** (the product's own suggestions) is scored with the
  identical metric on the identical pairs, so it is a fair comparison and not
  a strawman.

Embeddings are cached on disk; re-scoring is free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

import metrics as M
from engine import PROJECT, load_env

RUNS = Path(__file__).resolve().parent / "runs"
EMB_DB = RUNS / "embeddings.sqlite"
EMB_MODEL = "qwen3.7-text-embedding"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BATCH = 10


class EmbeddingStore:
    def __init__(self, model: str = EMB_MODEL):
        self.model = model
        EMB_DB.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(EMB_DB))
        self._con.execute("CREATE TABLE IF NOT EXISTS emb "
                          "(k TEXT PRIMARY KEY, v BLOB)")
        self._con.commit()
        self._client = None

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{text}".encode("utf-8")).hexdigest()

    def _client_or_die(self):
        if self._client is None:
            import os
            from openai import OpenAI
            load_env()
            key = os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                sys.exit("DASHSCOPE_API_KEY not set")
            self._client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60)
        return self._client

    def embed(self, texts: list[str]) -> dict[str, np.ndarray]:
        texts = [t for t in {t.strip() for t in texts} if t]
        out: dict[str, np.ndarray] = {}
        missing = []
        for t in texts:
            row = self._con.execute("SELECT v FROM emb WHERE k=?",
                                    (self._key(t),)).fetchone()
            if row:
                out[t] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(t)
        if missing:
            client = self._client_or_die()
            print(f"  embedding {len(missing)} new texts "
                  f"({len(texts) - len(missing)} cached)...", flush=True)
            for i in range(0, len(missing), BATCH):
                chunk = [t[:2000] for t in missing[i:i + BATCH]]
                resp = client.embeddings.create(model=self.model, input=chunk)
                for text, item in zip(missing[i:i + BATCH], resp.data):
                    vec = np.asarray(item.embedding, dtype=np.float32)
                    out[text] = vec
                    self._con.execute("INSERT OR REPLACE INTO emb VALUES (?,?)",
                                      (self._key(text), vec.tobytes()))
                self._con.commit()
        return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def load_records(run_id: str) -> list[dict]:
    path = RUNS / run_id / "predictions.jsonl"
    if not path.exists():
        sys.exit(f"no predictions at {path}")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--calibrate", action="store_true",
                    help="print the similarity distribution and exit")
    args = ap.parse_args()

    records = [r for r in load_records(args.run) if r.get("parse_ok")]
    if not records:
        sys.exit("no parsed records")

    store = EmbeddingStore()
    texts: list[str] = []
    for r in records:
        texts.append(str(r["reference_next"]))
        texts.extend(str(s) for s in (r["pred"] or []))
        texts.extend(str(s) for s in (r.get("deployed_suggestions_list") or []))
    vecs = store.embed(texts)

    def sims(sugg: list, ref: str) -> list[float]:
        rv = vecs.get(ref.strip())
        if rv is None:
            return []
        return [_cos(vecs[s.strip()], rv) for s in sugg
                if s and s.strip() in vecs]

    by_cond: dict[str, list] = {}
    per_user: dict[str, dict[str, list]] = {}
    for r in records:
        s = sims(r["pred"] or [], str(r["reference_next"]))
        by_cond.setdefault(r["condition"], []).append(s)
        per_user.setdefault(r["condition"], {}).setdefault(r["user_id"], []).append(s)

    if args.calibrate:
        allsim = [x for v in by_cond.values() for s in v for x in s]
        a = np.asarray(allsim)
        print(f"\nsimilarity distribution over {len(a)} pairs")
        for q in (50, 75, 90, 95, 99):
            print(f"  p{q}: {np.percentile(a, q):.3f}")
        print("\npass@3 at candidate tau:")
        for tau in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            rates = {c: M.semantic_pass_at_k(v, tau)["pass@3"]
                     for c, v in by_cond.items()}
            print(f"  tau={tau:.2f}  " + "  ".join(
                f"{c}={x:.3f}" for c, x in sorted(rates.items())))
        return

    report = {"run": args.run, "tau": args.tau, "embedding_model": EMB_MODEL,
              "n_records": len(records), "conditions": {}}
    for cond, v in sorted(by_cond.items()):
        res = M.semantic_pass_at_k(v, args.tau)
        preds = [r["pred"] for r in records if r["condition"] == cond]
        res["distinct_2"] = M.distinct_n(preds)
        res["n"] = len(v)
        pu = {u: M.semantic_pass_at_k(vv, args.tau)["pass@3"]
              for u, vv in per_user[cond].items()}
        res["per_user_pass@3"] = pu
        res["user_macro_pass@3"] = round(float(np.mean(list(pu.values()))), 4)
        lo, hi = M.bootstrap_ci(list(pu.values()))
        res["user_macro_ci95"] = [lo, hi]
        report["conditions"][cond] = res

    # the deployed system, scored identically on the same pairs
    dep = [(r["deployed_suggestions_list"] or [], str(r["reference_next"]))
           for r in records if r.get("deployed_suggestions_list")]
    if dep:
        dv = [sims(s, ref) for s, ref in dep]
        report["deployed_baseline"] = M.semantic_pass_at_k(dv, args.tau)
        report["deployed_baseline"]["n"] = len(dv)
        report["deployed_baseline"]["note"] = (
            "the product's own suggestions, same metric, same pairs")

    # paired tests: every condition against every control
    controls = [c for c in by_cond if c.startswith("ctrl_") or c == "b0_none"]
    comps = {}
    for cond in by_cond:
        if cond in controls:
            continue
        for ref in controls:
            a = report["conditions"][cond]["per_user_pass@3"]
            b = report["conditions"][ref]["per_user_pass@3"]
            comps[f"{cond}__vs__{ref}"] = M.paired_wilcoxon(a, b)
    report["paired_comparisons"] = comps

    out = RUNS / args.run / "semantic_scores.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\ntau={args.tau}  n={len(records)}  model={EMB_MODEL}")
    print(f"{'condition':<20}{'pass@1':>8}{'pass@3':>8}{'MRR':>8}"
          f"{'macro@3':>9}{'dist2':>8}")
    for cond, res in sorted(report["conditions"].items(),
                            key=lambda kv: -kv[1]["pass@3"]):
        print(f"{cond:<20}{res['pass@1']:>8.3f}{res['pass@3']:>8.3f}"
              f"{res['mrr']:>8.3f}{res['user_macro_pass@3']:>9.3f}"
              f"{res['distinct_2']:>8.3f}")
    if "deployed_baseline" in report:
        d = report["deployed_baseline"]
        print(f"{'[deployed system]':<20}{d['pass@1']:>8.3f}{d['pass@3']:>8.3f}"
              f"{d['mrr']:>8.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
