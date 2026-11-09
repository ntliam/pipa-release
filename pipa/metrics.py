"""Metrics, chosen for the specific pathologies of this data.

  * AUPRC for production timing - base rate is 12.3%, so AUROC flatters.
  * user-macro everywhere - Clement is 38% of the annotated corpus.
  * FPR at fixed recall - proactive interruption has asymmetric cost.
  * macro-F1 for intention - classes span 565 to 17 samples.
  * paired Wilcoxon across users for condition deltas - n = 13-26 users.
"""
from __future__ import annotations

import collections
import math
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# binary
# --------------------------------------------------------------------------


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool],
                   scores: Sequence[float] | None = None) -> dict:
    y = np.asarray(list(y_true), dtype=bool)
    p = np.asarray(list(y_pred), dtype=bool)
    if len(y) == 0:
        return {"n": 0}
    tp = int((y & p).sum()); fp = int((~y & p).sum())
    fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(float((y == p).mean()), 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        # false alarms per non-need moment: the interruption cost
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if scores is not None and len(set(map(float, scores))) > 1:
        s = np.asarray(list(scores), dtype=float)
        out["auroc"] = round(_auroc(y, s), 4)
        out["auprc"] = round(_auprc(y, s), 4)
        out["fpr_at_recall_50"] = round(_fpr_at_recall(y, s, 0.5), 4)
    return out


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision - the step-wise estimator, no interpolation."""
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = int(y.sum())
    return float((prec * y).sum() / n_pos) if n_pos else float("nan")


def _fpr_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    order = np.argsort(-s)
    ys = y[order]
    n_pos, n_neg = int(ys.sum()), int((~ys).sum())
    if not n_pos or not n_neg:
        return float("nan")
    tp = fp = 0
    for label in ys:
        tp += bool(label); fp += (not label)
        if tp / n_pos >= target:
            return fp / n_neg
    return 1.0


# --------------------------------------------------------------------------
# multiclass
# --------------------------------------------------------------------------


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                       labels: Sequence[str] | None = None) -> dict:
    y = list(y_true); p = list(y_pred)
    if not y:
        return {"n": 0}
    labels = list(labels) if labels else sorted(set(y) | set(p))
    per = {}
    f1s, precs, recs = [], [], []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(1 for a in y if a == c)
        if support:                      # ignore classes absent from the truth
            f1s.append(f1); precs.append(prec); recs.append(rec)
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "support": support}
    return {
        "n": len(y),
        "accuracy": round(sum(a == b for a, b in zip(y, p)) / len(y), 4),
        "macro_precision": round(float(np.mean(precs)), 4) if precs else 0.0,
        "macro_recall": round(float(np.mean(recs)), 4) if recs else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        # a model that answers one class for everything gets macro_f1 ~ 0.02 but
        # can still look reasonable on accuracy - report both.
        "n_classes_predicted": len({b for b in p}),
        "per_class": per,
    }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def semantic_pass_at_k(sims_per_item: Sequence[Sequence[float]], tau: float,
                       ks=(1, 3, 5)) -> dict:
    """sims_per_item[i][j] = similarity of suggestion j to what the user did.

    Suggestions must already be in the model's own ranked order.
    """
    out = {}
    for k in ks:
        hits, best = [], []
        for sims in sims_per_item:
            top = list(sims)[:k]
            hits.append(max(top) >= tau if top else False)
            best.append(max(top) if top else 0.0)
        out[f"pass@{k}"] = round(float(np.mean(hits)), 4) if hits else 0.0
        out[f"mean_max_sim@{k}"] = round(float(np.mean(best)), 4) if best else 0.0
    rr = []
    for sims in sims_per_item:
        rank = next((i + 1 for i, s in enumerate(sims) if s >= tau), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = round(float(np.mean(rr)), 4) if rr else 0.0
    out["tau"] = tau
    return out


def distinct_n(suggestion_lists: Sequence[Sequence[str]], n: int = 2) -> float:
    """Guards against three paraphrases of one safe suggestion."""
    total, uniq = 0, set()
    for lst in suggestion_lists:
        for s in lst:
            toks = list(s)
            for i in range(max(0, len(toks) - n + 1)):
                uniq.add(tuple(toks[i:i + n]))
                total += 1
    return round(len(uniq) / total, 4) if total else 0.0


# --------------------------------------------------------------------------
# user-macro and paired testing
# --------------------------------------------------------------------------


def user_macro(per_user: dict[str, dict], key: str,
               min_n: int = 1) -> dict:
    """Average a metric across users, ignoring users where it is undefined.

    Degenerate users are real here: one annotated user has 0 positives and
    another has 0 negatives, so their F1 is undefined rather than zero.
    """
    vals, skipped = [], []
    for user, m in per_user.items():
        v = m.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)) or m.get("n", 0) < min_n:
            skipped.append(user)
            continue
        if key == "f1" and m.get("positive_rate") in (0.0, 1.0):
            skipped.append(user)     # undefined, not zero
            continue
        vals.append(float(v))
    return {
        f"user_macro_{key}": round(float(np.mean(vals)), 4) if vals else None,
        "users_counted": len(vals),
        "users_skipped": skipped,
    }


def paired_wilcoxon(a: dict[str, float], b: dict[str, float]) -> dict:
    """Paired signed-rank across users. With n=13-26 users this is the honest
    test; a between-group test at that n is not interpretable."""
    users = sorted(set(a) & set(b))
    diffs = [a[u] - b[u] for u in users
             if a[u] is not None and b[u] is not None]
    diffs = [d for d in diffs if not math.isnan(d)]
    n_all = len(diffs)
    nz = [d for d in diffs if d != 0]
    res = {
        "n_users": n_all, "n_nonzero": len(nz),
        "mean_delta": round(float(np.mean(diffs)), 4) if diffs else None,
        "median_delta": round(float(np.median(diffs)), 4) if diffs else None,
        "n_positive": sum(d > 0 for d in diffs),
    }
    if len(nz) < 6:
        res["p_value"] = None
        res["note"] = "too few non-zero pairs for a meaningful test"
        return res
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(nz)
        res["statistic"], res["p_value"] = float(stat), round(float(p), 5)
    except ImportError:
        ranks = np.argsort(np.argsort([abs(d) for d in nz])) + 1
        w = sum(r for r, d in zip(ranks, nz) if d > 0)
        n = len(nz)
        mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w - mu) / sd if sd else 0.0
        res["statistic"] = float(w)
        res["p_value"] = round(2 * (1 - _norm_cdf(abs(z))), 5)
        res["note"] = "normal approximation (scipy unavailable)"
    return res


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> tuple:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], float)
    if len(v) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return (round(float(np.percentile(means, 100 * alpha / 2)), 4),
            round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4))
