"""Table: effect of uncertainty-aware intervention, on REAL held-out decisions.

    python bayes_table.py                 # timing_annotated, lambda=0.95
    python bayes_table.py --lam 1.0       # stationary posterior, for contrast

Three arms, same posterior machinery, one thing removed each time:

  PIPA          personalised posterior, updated online, decides from posterior
                MASS (act / ask / silent).
  PIPA-Mean     same posterior, collapsed to E[theta]; can never ask.
  PIPA (k=0)    NO user-specific evidence - the population prior only, never
                updated. Keeps the mass-based policy. Isolates personalisation
                from the decision rule.

Sequential protocol (the one that survived E5): prior fitted on the first half
of every user's history; then walk each user's SECOND half, decide before each
event, observe, update with forgetting  a <- lam*a + y,  b <- lam*b + (1-y).
Every decision is scored against a label the model has not yet seen.

Metric definitions, stated because `ask` is a third action:
  positive prediction  = act OR ask     (help is offered either way)
  false alarm          = act AND not wanted   (an ask is not an alarm)
  F1 / Acc             on positive-vs-negative as above
  Utility              mean of Costs.utility(action, wanted), the same costs
                       that derive the decision thresholds
Confidence intervals: paired block bootstrap over USERS, so users with long
histories do not dominate, and the PIPA-minus-PIPA-Mean interval is on the
DIFFERENCE from the same draws.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Force UTF-8 on stdout. reconfigure() is idempotent; re-wrapping with
# TextIOWrapper closes the shared buffer when a second module does the
# same in one process, which breaks any script that imports two of these.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
EXPORTS = Path(__file__).resolve().parent / "exports"

from bayes import BetaPosterior, Costs, decide, decide_mean, fit_prior  # noqa: E402

COSTS = Costs()
EPS = 0.2
ARMS = ("PIPA-Mean", "PIPA (k=0)", "PIPA")


def metrics(actions, y):
    a = np.array(actions); yy = np.array(y, dtype=bool)
    pos = (a == "act") | (a == "ask")
    tp = int((pos & yy).sum()); fp = int((pos & ~yy).sum())
    fn = int((~pos & yy).sum()); tn = int((~pos & ~yy).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    neg = int((~yy).sum())
    far = float(((a == "act") & ~yy).sum() / neg) if neg else 0.0
    util = float(np.mean([COSTS.utility(ac, bool(w)) for ac, w in zip(actions, y)]))
    return {"f1": f1, "far": far, "acc": (tp + tn) / len(yy), "utility": util,
            "ask_rate": float((a == "ask").mean()), "act_rate": float((a == "act").mean()),
            "n": int(len(yy))}


def run(task, lam, min_events):
    import tasks
    ts = tasks.build(task)
    per_user = collections.defaultdict(list)
    for inst in ts.instances:
        lab = inst.target.get("label")
        if isinstance(lab, bool):
            per_user[inst.user_id].append((inst.t0, lab))
    users = {u: sorted(v) for u, v in per_user.items() if len(v) >= min_events}
    first = {u: [l for _, l in v[: len(v) // 2]] for u, v in users.items()}
    second = {u: [l for _, l in v[len(v) // 2:]] for u, v in users.items()}
    rates = np.array([np.mean(f) for f in first.values()])
    counts = np.array([len(f) for f in first.values()], dtype=float)
    a0, b0 = fit_prior(rates, counts)
    prior = BetaPosterior.prior(a0, b0)

    per = {arm: {} for arm in ARMS}
    labels = {}
    for u in users:
        f, sec = first[u], second[u]
        a, b = prior.a + sum(f), prior.b + (len(f) - sum(f))
        acts = {arm: [] for arm in ARMS}
        for y in sec:
            post = BetaPosterior(a, b)
            acts["PIPA"].append(decide(post, COSTS, EPS))
            acts["PIPA-Mean"].append(decide_mean(post, COSTS))
            acts["PIPA (k=0)"].append(decide(prior, COSTS, EPS))
            a, b = lam * a + y, lam * b + (1 - y)
        for arm in ARMS:
            per[arm][u] = acts[arm]
        labels[u] = list(sec)

    ulist = sorted(users)
    pooled = {arm: metrics([x for u in ulist for x in per[arm][u]],
                           [x for u in ulist for x in labels[u]]) for arm in ARMS}

    rng = np.random.default_rng(20260917)
    boots = {arm: collections.defaultdict(list) for arm in ARMS}
    for _ in range(3000):
        draw = rng.choice(ulist, size=len(ulist), replace=True)
        for arm in ARMS:
            m = metrics([x for u in draw for x in per[arm][u]],
                        [x for u in draw for x in labels[u]])
            for k in ("f1", "far", "acc", "utility"):
                boots[arm][k].append(m[k])
    ci = {arm: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for k, v in boots[arm].items()} for arm in ARMS}
    paired = {}
    for k in ("f1", "far", "acc", "utility"):
        d = np.array(boots["PIPA"][k]) - np.array(boots["PIPA-Mean"][k])
        paired[k] = {"delta": pooled["PIPA"][k] - pooled["PIPA-Mean"][k],
                     "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "p_le_0": float(np.mean(d <= 0)), "p_ge_0": float(np.mean(d >= 0))}
    return {"task": task, "lambda": lam, "n_users": len(ulist),
            "prior": [round(a0, 3), round(b0, 3)], "eps": EPS, "costs": COSTS.__dict__,
            "thresholds": list(COSTS.thresholds()), "pooled": pooled, "ci95": ci,
            "paired_PIPA_minus_Mean": paired}


def latex(r):
    """Bold marks the best value in each COLUMN. One row does not win
    everything, and the table must not say that it does."""
    p = r["pooled"]
    best = {"f1": max(ARMS, key=lambda a: p[a]["f1"]),
            "far": min(ARMS, key=lambda a: p[a]["far"]),
            "acc": max(ARMS, key=lambda a: p[a]["acc"]),
            "utility": max(ARMS, key=lambda a: p[a]["utility"])}
    B = "\\textbf{"

    def cell(arm, k):
        v = p[arm][k]
        txt = f"{v:+.3f}" if k == "utility" else f"{100 * v:.1f}"
        return B + txt + "}" if best[k] == arm else txt

    names = {"PIPA-Mean": "PIPA-Mean", "PIPA (k=0)": "PIPA ($\\kappa=0$)", "PIPA": B + "PIPA}"}
    rows = [f"{names[a]} & {cell(a, 'f1')} & {cell(a, 'far')} & {cell(a, 'acc')} & {cell(a, 'utility')} \\\\"
            for a in ARMS]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="timing_annotated")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--min-events", type=int, default=40)
    args = ap.parse_args()
    r = run(args.task, args.lam, args.min_events)

    print(f"# {r['task']}  users={r['n_users']}  lambda={r['lambda']}  "
          f"prior Beta{tuple(r['prior'])}  tau=({r['thresholds'][0]:.3f}, {r['thresholds'][1]:.3f})\n")
    print(f"{'Method':12}{'F1':>7}{'FAR':>7}{'Acc':>7}{'Utility':>9}{'ask%':>7}{'act%':>7}{'n':>6}")
    print("-" * 62)
    for arm in ARMS:
        m = r["pooled"][arm]
        print(f"{arm:12}{m['f1']:7.3f}{m['far']:7.3f}{m['acc']:7.3f}{m['utility']:9.3f}"
              f"{100 * m['ask_rate']:7.1f}{100 * m['act_rate']:7.1f}{m['n']:6}")
    print("\nPaired PIPA - PIPA-Mean (block bootstrap over users):")
    for k, v in r["paired_PIPA_minus_Mean"].items():
        lo, hi = v["ci95"]
        sig = "  <-- CI excludes 0" if not (lo <= 0 <= hi) else ""
        print(f"  {k:8} {v['delta']:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")

    out = EXPORTS / f"bayes_table_{args.task}_lam{args.lam}.json"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"\nwrote {out}\n\n--- LaTeX rows (bold = best per column) ---")
    print(latex(r))


if __name__ == "__main__":
    main()
