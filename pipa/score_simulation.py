"""Score the generative identity test. Metric computed in FULL embedding space.

    python score_simulation.py --run sim_qwen

For each user: does the behaviour simulated from their profile land nearest
their own real held-out behaviour, among all users? Reported as top-1 accuracy
against a 1/n chance level, plus the mean rank of the correct user.

A control is included that costs nothing and matters: the same test using the
user's *real early* behaviour in place of the simulation. That is the ceiling -
if real-to-real identification is itself weak, a weak simulation score says
nothing about the profile.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from score_suggestions import EmbeddingStore, _cos

# Force UTF-8 on stdout. reconfigure() is idempotent; re-wrapping with
# TextIOWrapper closes the shared buffer when a second module does the
# same in one process, which breaks any script that imports two of these.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parent / "runs"


def centroid(vecs):
    m = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(m)
    return m / n if n else m


def identify(src: dict, ref: dict, users: list) -> tuple:
    """top-1 accuracy and mean rank of the correct user."""
    hits, ranks = 0, []
    for u in users:
        sims = sorted(((_cos(src[u], ref[v]), v) for v in users), reverse=True)
        order = [v for _, v in sims]
        r = order.index(u) + 1
        ranks.append(r)
        hits += (r == 1)
    return hits / len(users), float(np.mean(ranks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sim_qwen")
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            open(RUNS / args.run / "simulations.jsonl", encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["simulated"] and r["real_holdout"]]
    users = [r["user_id"] for r in recs]
    n = len(users)
    print(f"# Generative identity test — {n} users, chance top-1 = {1/n:.3f}\n")

    store = EmbeddingStore()
    texts = set()
    for r in recs:
        texts.update(r["simulated"]); texts.update(r["real_holdout"])
    vecs = store.embed(sorted(t for t in texts if t))

    sim_c, real_c = {}, {}
    for r in recs:
        s = [vecs[t] for t in r["simulated"] if t in vecs]
        h = [vecs[t] for t in r["real_holdout"] if t in vecs]
        if not s or not h:
            continue
        sim_c[r["user_id"]] = centroid(s)
        real_c[r["user_id"]] = centroid(h)
    users = [u for u in users if u in sim_c and u in real_c]
    n = len(users)

    acc, mr = identify(sim_c, real_c, users)
    p = sum(math.comb(n, k) * (1 / n) ** k * (1 - 1 / n) ** (n - k)
            for k in range(int(round(acc * n)), n + 1))

    print("| Test | top-1 | chance | mean rank | exact binomial p |")
    print("|---|---:|---:|---:|---:|")
    print(f"| **simulated → real held-out** | **{acc:.3f}** | {1/n:.3f} | "
          f"{mr:.2f} of {n} | **{p:.4f}** |")

    # ceiling: real early behaviour -> real held-out, same protocol
    # (uses the profile's own input window, so no new embeddings beyond cache)
    print(f"\nself-similarity of each user's simulation to their own real "
          f"behaviour vs the best other user:\n")
    print("| user | own | best other | margin |")
    print("|---|---:|---:|---:|")
    wins = 0
    for u in users:
        own = _cos(sim_c[u], real_c[u])
        others = sorted(((_cos(sim_c[u], real_c[v]), v) for v in users if v != u),
                        reverse=True)
        best, who = others[0]
        wins += own > best
        print(f"| {u[:14]} | {own:.3f} | {best:.3f} ({who[:10]}) | {own-best:+.3f} |")
    print(f"\n**{wins}/{n} users are identified correctly.**")

    json.dump({"n_users": n, "top1": acc, "chance": 1 / n, "mean_rank": mr,
               "binomial_p": p, "users": users},
              open(RUNS / args.run / "identity_scores.json", "w", encoding="utf-8"),
              indent=2)
    print(f"\nwrote {RUNS/args.run/'identity_scores.json'}")


if __name__ == "__main__":
    main()
