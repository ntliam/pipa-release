"""The four experiments the Bayesian claim needs. Synthetic where theta* must be
known; real data where it can be.

    python bayes_experiments.py            # writes exports/bayes_results.json
    python bayes_experiments.py --seed 7

E1  CALIBRATION        Synthetic users with known theta*. For each history
                       length K: posterior MAE, Brier, ECE, and 95% credible-
                       interval COVERAGE P(theta* in CI). A calibrated posterior
                       covers ~0.95; a point estimate has no interval to test.

E2  CONVERGENCE        Var(theta | H) and CI width against K. The claim is
                       K up => Var down, with decisions becoming more decisive
                       (fewer "ask") as evidence accumulates.

E3  PIPA vs PIPA-MEAN  Identical posteriors; one keeps the spread, one throws
                       it away. Simulated proactive decisions under asymmetric
                       costs. Compare utility, false-alarm rate, regret vs an
                       oracle that knows theta*. Reported per K, because the
                       spread only matters when it is wide.

E4  SAME MEAN,         U_A saw 7/10 positive, U_B saw 70/100. E[theta] is the
    DIFFERENT EVIDENCE same to two decimals; Var differs tenfold. PIPA-Mean must
                       treat them identically. PIPA asks U_A and acts for U_B.

E5  REAL DATA          The same posterior on timing_annotated: prior fitted on
                       the population, posterior from each user's first half,
                       coverage of the HELD-OUT rate. No theta* exists, so this
                       is the weaker test - but it is the one on real people.

Nothing is tuned to a result. eps (the ask band), tau and the costs are fixed
before running and are the same for every arm.
"""
from __future__ import annotations

import argparse
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

from bayes import (BetaPosterior, Costs, decide, decide_mean,      # noqa: E402
                   decide_oracle, fit_prior)

EPS = 0.2
KS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]
COSTS = Costs()
TAU_SILENT, TAU_ACT = COSTS.thresholds()


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out, n = 0.0, len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            out += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(out)


# ------------------------------------------------------------ synthetic
def synth_population(rng, n_users: int, a_true: float, b_true: float):
    """Users are heterogeneous: theta* ~ Beta(a_true, b_true)."""
    return rng.beta(a_true, b_true, size=n_users)


def e1_e2_calibration_convergence(rng, n_users=400, a_true=2.0, b_true=2.5):
    theta = synth_population(rng, n_users, a_true, b_true)
    # the agent does NOT know a_true/b_true: it fits its prior from a small
    # pilot of observed rates, the way it would in deployment
    pilot_k = rng.binomial(20, theta[:50])
    a0, b0 = fit_prior(pilot_k / 20, np.full(50, 20.0))
    prior = BetaPosterior.prior(a0, b0)
    rows = []
    for K in KS:
        k = rng.binomial(K, theta)
        posts = [prior.update(int(ki), K) for ki in k]
        mean = np.array([p.mean for p in posts])
        var = np.array([p.var for p in posts])
        cis = np.array([p.ci(0.95) for p in posts])
        width = cis[:, 1] - cis[:, 0]
        cover = float(((theta >= cis[:, 0]) & (theta <= cis[:, 1])).mean())
        y_next = rng.binomial(1, theta)          # one fresh outcome per user
        rows.append({
            "K": K,
            "posterior_mae": float(np.abs(mean - theta).mean()),
            "brier": float(((mean - y_next) ** 2).mean()),
            "ece": ece(mean, y_next.astype(float)),
            "coverage_95": cover,
            "mean_var": float(var.mean()),
            "mean_ci_width": float(width.mean()),
            "frac_ask": float(np.mean([decide(p, COSTS, EPS) == "ask" for p in posts])),
        })
    return {"prior_fitted": [round(a0, 3), round(b0, 3)],
            "prior_true": [a_true, b_true], "n_users": n_users, "rows": rows}


def e3_pipa_vs_mean(rng, n_users=400, a_true=2.0, b_true=2.5, n_decisions=30):
    """Each user faces n_decisions moments; outcome ~ Bernoulli(theta*)."""
    theta = synth_population(rng, n_users, a_true, b_true)
    pilot_k = rng.binomial(20, theta[:50])
    a0, b0 = fit_prior(pilot_k / 20, np.full(50, 20.0))
    prior = BetaPosterior.prior(a0, b0)
    rows = []
    for K in KS:
        k = rng.binomial(K, theta)
        posts = [prior.update(int(ki), K) for ki in k]
        stat = {arm: {"utility": 0.0, "fa": 0, "neg": 0, "ask": 0, "n": 0}
                for arm in ("pipa", "pipa_mean", "oracle")}
        for u, p in enumerate(posts):
            wanted = rng.binomial(1, theta[u], size=n_decisions).astype(bool)
            acts = {"pipa": decide(p, COSTS, EPS), "pipa_mean": decide_mean(p, COSTS),
                    "oracle": decide_oracle(theta[u], COSTS)}
            for arm, a in acts.items():
                s = stat[arm]
                for w in wanted:
                    s["utility"] += COSTS.utility(a, bool(w))
                    s["n"] += 1
                    if not w:
                        s["neg"] += 1
                        s["fa"] += (a == "act")
                    s["ask"] += (a == "ask")
        out = {"K": K}
        for arm, s in stat.items():
            out[arm] = {"utility": s["utility"] / s["n"],
                        "far": s["fa"] / max(s["neg"], 1),
                        "ask_rate": s["ask"] / s["n"]}
        out["regret_pipa"] = out["oracle"]["utility"] - out["pipa"]["utility"]
        out["regret_mean"] = out["oracle"]["utility"] - out["pipa_mean"]["utility"]
        rows.append(out)
    return {"tau_silent": TAU_SILENT, "tau_act": TAU_ACT, "eps": EPS,
            "costs": COSTS.__dict__, "rows": rows}


def e4_same_mean(a0=1.0, b0=1.0):
    """Beta(1,1) prior makes the arithmetic transparent; the point survives any
    proper prior because the MEAN converges while the VARIANCE shrinks."""
    prior = BetaPosterior.prior(a0, b0)
    A = prior.update(7, 10)
    B = prior.update(70, 100)
    out = {}
    for name, p in (("U_A (7/10)", A), ("U_B (70/100)", B)):
        lo, hi = p.ci(0.95)
        out[name] = {"mean": round(p.mean, 4), "var": round(p.var, 5),
                     "ci95": [round(lo, 3), round(hi, 3)],
                     "P(theta>tau_act)": round(p.prob_above(TAU_ACT), 4),
                     "PIPA": decide(p, COSTS, EPS), "PIPA-Mean": decide_mean(p, COSTS)}
    out["_var_ratio_A_over_B"] = round(A.var / B.var, 2)
    return out


# ------------------------------------------------------------- real data
def e5_real(task="timing_annotated"):
    import collections
    import tasks
    ts = tasks.build(task)
    per_user = collections.defaultdict(list)
    for inst in ts.instances:
        lab = inst.target.get("label")
        if isinstance(lab, bool):
            per_user[inst.user_id].append((inst.t0, lab))
    users = {u: sorted(v) for u, v in per_user.items() if len(v) >= 40}
    # prior from FIRST halves only - the held-out halves never touch the prior
    first = {u: [l for _, l in v[: len(v) // 2]] for u, v in users.items()}
    second = {u: [l for _, l in v[len(v) // 2:]] for u, v in users.items()}
    rates = np.array([np.mean(f) for f in first.values()])
    counts = np.array([len(f) for f in first.values()], dtype=float)
    a0, b0 = fit_prior(rates, counts)
    prior = BetaPosterior.prior(a0, b0)
    rows, cover = [], 0
    for u in users:
        f, s = first[u], second[u]
        p = prior.update(int(sum(f)), len(f))
        lo, hi = p.ci(0.95)
        held = float(np.mean(s))
        inside = lo <= held <= hi
        cover += inside
        rows.append({"user": u, "n_first": len(f), "n_heldout": len(s),
                     "posterior_mean": round(p.mean, 4), "ci95": [round(lo, 3), round(hi, 3)],
                     "heldout_rate": round(held, 4), "covered": bool(inside),
                     "PIPA": decide(p, COSTS, EPS), "PIPA-Mean": decide_mean(p, COSTS)})
    # the same users, but the posterior built from only their first K events
    conv = []
    for K in (4, 8, 16, 32, 64):
        widths, errs, cov = [], [], 0
        for u in users:
            f = first[u][:K]
            if len(f) < K:
                continue
            p = prior.update(int(sum(f)), len(f))
            lo, hi = p.ci(0.95)
            held = float(np.mean(second[u]))
            widths.append(hi - lo); errs.append(abs(p.mean - held)); cov += lo <= held <= hi
        if widths:
            conv.append({"K": K, "n_users": len(widths), "mean_ci_width": float(np.mean(widths)),
                         "heldout_mae": float(np.mean(errs)), "coverage_95": cov / len(widths)})
    # E5b - drift. A stationary posterior tightens on the first half and then
    # misses the second: real users move. forget(gamma) discounts accumulated
    # evidence toward the prior scale. Sweep it and watch coverage.
    drift = []
    for gamma in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        cov, widths, errs = 0, [], []
        for u in users:
            f, s = first[u], second[u]
            p = prior.update(int(sum(f)), len(f)).forget(gamma)
            # forgetting must not shrink below the prior itself
            if p.a + p.b < prior.a + prior.b:
                p = prior.update(int(sum(f)), len(f))
                scale = (prior.a + prior.b) / (p.a + p.b)
                p = p.forget(scale)
            lo, hi = p.ci(0.95); held = float(np.mean(s))
            cov += lo <= held <= hi; widths.append(hi - lo); errs.append(abs(p.mean - held))
        drift.append({"gamma": gamma, "coverage_95": cov / len(users),
                      "mean_ci_width": float(np.mean(widths)),
                      "heldout_mae": float(np.mean(errs))})
    # E5c - ONLINE. The one-shot posterior is wrong about the second half
    # because users move. So walk the second half sequentially: before each
    # event predict, observe, update with exponential forgetting
    #     a <- lam*a + y,   b <- lam*b + (1-y)
    # and score the one-step-ahead prediction. lam=1 is the stationary model;
    # lam<1 has an effective memory of 1/(1-lam) events.
    online = []
    for lam in (1.0, 0.99, 0.97, 0.95, 0.90):
        P, Y, covs = [], [], []
        for u in users:
            f, sec = first[u], second[u]
            a, b = prior.a + sum(f), prior.b + (len(f) - sum(f))
            hits = 0
            for t, y in enumerate(sec):
                post = BetaPosterior(a, b)
                P.append(post.mean); Y.append(float(y))
                lo, hi = post.ci(0.95)
                # rolling truth: the rate over the next 20 events
                win = sec[t:t + 20]
                if len(win) >= 10:
                    hits += lo <= float(np.mean(win)) <= hi; covs.append(None)
                a, b = lam * a + y, lam * b + (1 - y)
            covs = [c for c in covs if c is None]
        P, Y = np.array(P), np.array(Y)
        # coverage of the rolling 20-event rate, pooled over users and steps
        cov_hits, cov_n = 0, 0
        for u in users:
            f, sec = first[u], second[u]
            a, b = prior.a + sum(f), prior.b + (len(f) - sum(f))
            for t, y in enumerate(sec):
                lo, hi = BetaPosterior(a, b).ci(0.95)
                win = sec[t:t + 20]
                if len(win) >= 10:
                    cov_hits += lo <= float(np.mean(win)) <= hi; cov_n += 1
                a, b = lam * a + y, lam * b + (1 - y)
        online.append({"lambda": lam, "eff_memory": (1 / (1 - lam)) if lam < 1 else float("inf"),
                       "brier": float(((P - Y) ** 2).mean()), "ece": ece(P, Y),
                       "rolling_coverage_95": cov_hits / max(cov_n, 1), "n_steps": int(len(Y))})
    return {"task": task, "n_users": len(users), "prior_fitted": [round(a0, 3), round(b0, 3)],
            "coverage_95": cover / len(users), "per_user": rows, "convergence": conv,
            "drift_sweep": drift, "online": online}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    res = {"seed": args.seed, "tau_silent": TAU_SILENT, "tau_act": TAU_ACT, "eps": EPS,
           "costs": COSTS.__dict__}
    res["E1_E2"] = e1_e2_calibration_convergence(rng)
    res["E3"] = e3_pipa_vs_mean(rng)
    res["E4"] = e4_same_mean()
    try:
        res["E5"] = e5_real()
    except Exception as exc:                                   # noqa: BLE001
        res["E5"] = {"error": f"{type(exc).__name__}: {exc}"}

    print("## E1/E2 synthetic — calibration and convergence")
    print(f"prior fitted {res['E1_E2']['prior_fitted']} (true {res['E1_E2']['prior_true']})")
    print(f"{'K':>4} {'MAE':>7} {'Brier':>7} {'ECE':>7} {'cover95':>8} {'Var':>8} {'CIwidth':>8} {'ask%':>6}")
    for r in res["E1_E2"]["rows"]:
        print(f"{r['K']:>4} {r['posterior_mae']:7.3f} {r['brier']:7.3f} {r['ece']:7.3f} "
              f"{r['coverage_95']:8.3f} {r['mean_var']:8.4f} {r['mean_ci_width']:8.3f} {100*r['frac_ask']:6.1f}")

    print("\n## E3 synthetic — PIPA vs PIPA-Mean")
    print(f"{'K':>4} | {'util PIPA':>9} {'util Mean':>9} {'oracle':>7} | {'FAR PIPA':>8} {'FAR Mean':>8} | {'ask%':>5} | {'regret P':>8} {'regret M':>8}")
    for r in res["E3"]["rows"]:
        print(f"{r['K']:>4} | {r['pipa']['utility']:9.3f} {r['pipa_mean']['utility']:9.3f} {r['oracle']['utility']:7.3f} | "
              f"{r['pipa']['far']:8.3f} {r['pipa_mean']['far']:8.3f} | {100*r['pipa']['ask_rate']:5.1f} | "
              f"{r['regret_pipa']:8.3f} {r['regret_mean']:8.3f}")

    print("\n## E4 — same mean, different evidence")
    for k, v in res["E4"].items():
        print(f"  {k}: {v}")

    if "error" not in res["E5"]:
        e5 = res["E5"]
        print(f"\n## E5 real ({e5['task']}) — {e5['n_users']} users, prior {e5['prior_fitted']}")
        print(f"held-out rate inside 95% CI: {e5['coverage_95']:.3f}")
        acts = [r["PIPA"] for r in e5["per_user"]]
        print(f"PIPA actions: act {acts.count('act')}, ask {acts.count('ask')}, silent {acts.count('silent')}")
        for c in e5["convergence"]:
            print(f"  K={c['K']:>3} users={c['n_users']:>2} CI width {c['mean_ci_width']:.3f} "
                  f"held-out MAE {c['heldout_mae']:.3f} coverage {c['coverage_95']:.3f}")
        print("  drift sweep (forgetting factor gamma):")
        for d in e5["drift_sweep"]:
            print(f"    gamma={d['gamma']:<5} coverage {d['coverage_95']:.3f}  CI width {d['mean_ci_width']:.3f}  MAE {d['heldout_mae']:.3f}")
        print("  online sequential update over the held-out half (one-step-ahead):")
        for o in e5["online"]:
            mem = "inf" if o["eff_memory"] == float("inf") else f"{o['eff_memory']:.0f}"
            print(f"    lambda={o['lambda']:<5} memory~{mem:>4} events  Brier {o['brier']:.4f}  ECE {o['ece']:.4f}  rolling-cov {o['rolling_coverage_95']:.3f}  (n={o['n_steps']})")
    else:
        print("\n## E5 skipped:", res["E5"]["error"])

    EXPORTS.mkdir(exist_ok=True)
    out = EXPORTS / "bayes_results.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
