"""Table: how uncertainty changes intervention behaviour (action distribution).

    python action_table.py

Same decisions as bayes_table.py - real held-out events, sequential posterior
with forgetting - but reporting WHERE the decisions go rather than how well
they score.

Arms:

  PIPA-Mean          posterior collapsed to E[theta], action = argmax expected
                     utility at that point.
  PIPA-Mean-Matched  the control that matters for a distribution table. PIPA
                     intervenes less often than PIPA-Mean, so a lower false-alarm
                     rate could be nothing more than lower volume. This arm
                     thresholds the posterior MEAN so its act-rate equals PIPA's
                     exactly; any remaining difference is selection quality, not
                     volume. Act-rate is matched BY CONSTRUCTION - that column is
                     not a result for this row, it is the control being applied.
  PIPA (k=0)         population prior only, never updated; keeps the policy.
  PIPA               posterior mass past cost-derived thresholds.

Cbar is the mean cost per decision, -E[utility] under the same cost model that
derives the thresholds. Lower is better, and it is the single number that
trades off "intervened wrongly" against "stayed silent when wanted".

NOTE ON THE ACTION SPACE: there are three actions, not four. `suggest` (a
softer intervention between ask and act) is not implemented; adding it would
change every threshold and every previously reported number, so it is left out
rather than faked with a zero column.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
EXPORTS = Path(__file__).resolve().parent / "exports"

from bayes import BetaPosterior, Costs, decide, decide_mean, fit_prior  # noqa: E402

COSTS = Costs()
EPS = 0.2
ARMS = ("PIPA-Mean", "PIPA-Mean-Matched", "PIPA (k=0)", "PIPA")


def summarise(actions, y):
    a = np.array(actions)
    cost = -float(np.mean([COSTS.utility(ac, bool(w)) for ac, w in zip(actions, y)]))
    neg = int((~np.array(y, dtype=bool)).sum())
    far = float(((a == "act") & ~np.array(y, dtype=bool)).sum() / neg) if neg else 0.0
    return {"silent": float((a == "silent").mean()), "ask": float((a == "ask").mean()),
            "act": float((a == "act").mean()), "cbar": cost, "far": far, "n": len(a)}


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
    a0, b0 = fit_prior(np.array([np.mean(f) for f in first.values()]),
                       np.array([len(f) for f in first.values()], float))
    prior = BetaPosterior.prior(a0, b0)

    acts = {arm: [] for arm in ARMS}
    means, labels = [], []
    for u in sorted(users):
        f, sec = first[u], second[u]
        a, b = prior.a + sum(f), prior.b + (len(f) - sum(f))
        for y in sec:
            post = BetaPosterior(a, b)
            acts["PIPA"].append(decide(post, COSTS, EPS))
            acts["PIPA-Mean"].append(decide_mean(post, COSTS))
            acts["PIPA (k=0)"].append(decide(prior, COSTS, EPS))
            means.append(post.mean); labels.append(bool(y))
            a, b = lam * a + y, lam * b + (1 - y)

    # matched control: same act-rate as PIPA, chosen by posterior mean
    means = np.array(means)
    n_act = int((np.array(acts["PIPA"]) == "act").sum())
    n_sil = int((np.array(acts["PIPA"]) == "silent").sum())
    # Match the EXPENSIVE action only. Matching silent as well would force the
    # whole distribution to coincide with PIPA's and make the table tautological;
    # the question is where the remaining decisions land once act volume is held
    # equal, so silent/ask split on the mean-based threshold as PIPA-Mean would.
    tau_silent, _ = COSTS.thresholds()
    order = np.argsort(-means)
    matched = np.where(means < tau_silent, "silent", "ask").astype(object)
    matched[order[:n_act]] = "act"
    acts["PIPA-Mean-Matched"] = list(matched)

    return {"task": task, "lambda": lam, "n_users": len(users),
            "prior": [round(a0, 3), round(b0, 3)], "eps": EPS,
            "thresholds": list(COSTS.thresholds()), "costs": COSTS.__dict__,
            "rows": {arm: summarise(acts[arm], labels) for arm in ARMS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="timing_annotated")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--min-events", type=int, default=40)
    args = ap.parse_args()
    r = run(args.task, args.lam, args.min_events)
    R = r["rows"]

    print(f"# {r['task']}  users={r['n_users']}  lambda={r['lambda']}  "
          f"n={R['PIPA']['n']}  tau=({r['thresholds'][0]:.3f}, {r['thresholds'][1]:.3f})\n")
    print(f"{'Method':20}{'silent':>9}{'ask':>9}{'act':>9}{'Cbar':>9}{'FAR':>9}")
    print("-" * 65)
    for arm in ARMS:
        m = R[arm]
        print(f"{arm:20}{100*m['silent']:9.1f}{100*m['ask']:9.1f}{100*m['act']:9.1f}"
              f"{m['cbar']:9.3f}{m['far']:9.3f}")

    out = EXPORTS / f"action_distribution_{args.task}_lam{args.lam}.json"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    best_sil = max(ARMS, key=lambda a: R[a]["silent"])
    best_act = min(ARMS, key=lambda a: R[a]["act"])
    best_c = min(ARMS, key=lambda a: R[a]["cbar"])
    def c(arm, k, pct=True, best=None):
        v = R[arm][k] * (100 if pct else 1)
        t = f"{v:.1f}" if pct else f"{v:.3f}"
        return "\\textbf{" + t + "}" if best == arm else t
    names = {"PIPA-Mean": "PIPA-Mean", "PIPA-Mean-Matched": "PIPA-Mean-Matched",
             "PIPA (k=0)": "PIPA ($\\kappa=0$)", "PIPA": "\\textbf{PIPA}"}
    print("\n--- LaTeX rows ---")
    for arm in ARMS:
        act = c(arm, "act", best=best_act)
        if arm == "PIPA-Mean-Matched":
            act += "$^\\dagger$"
        print(f"{names[arm]} & {c(arm,'silent',best=best_sil)} & {c(arm,'ask')} & "
              f"{act} & {c(arm,'cbar',pct=False,best=best_c)} \\\\")


if __name__ == "__main__":
    main()
