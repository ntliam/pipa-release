"""Experiment runner.

    # inspect prompts, spend nothing (default)
    python run.py --task timing_annotated --conditions b0_none,b3_statistical --limit 20

    # verify credentials with one tiny call
    python run.py --healthcheck

    # real run, capped
    python run.py --task suggestion --live --budget 2.0 --limit 200

Every prediction is written to runs/<run_id>/predictions.jsonl together with
the exact prompt hash, so results are auditable after the fact.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import agent
import memory as mem
import metrics as M
import tasks
from engine import BudgetExceeded, LLMEngine, ProviderOutOfCredit
from react import ReactAgent

RUNS = Path(__file__).resolve().parent / "runs"


def _prompt_hash(messages) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def select_instances(ts: tasks.TaskSet, split: str, limit: int | None,
                     min_history: int, seed: int = 42):
    # "duplicate" marks instances kept only as history (see build_suggestion);
    # they must never be evaluated, including under --split all.
    pool = ([i for i in ts.instances if i.split != "duplicate"]
            if split == "all" else ts.split_of(split))
    pool = [i for i in pool if len(i.history_ids) >= min_history]
    pool.sort(key=lambda i: (i.user_id, i.t0))
    if limit and len(pool) > limit:
        # stratify by user so a cap does not silently become "the top user"
        by_user = collections.defaultdict(list)
        for i in pool:
            by_user[i.user_id].append(i)
        out, users = [], sorted(by_user)
        idx = 0
        while len(out) < limit:
            progressed = False
            for u in users:
                if idx < len(by_user[u]):
                    out.append(by_user[u][idx])
                    progressed = True
                    if len(out) >= limit:
                        break
            if not progressed:
                break
            idx += 1
        pool = sorted(out, key=lambda i: (i.user_id, i.t0))
    return pool


def pick_threshold(records: list[dict]) -> float:
    """Threshold maximising F1 on THIS set. Only ever call it on val/train
    records - calling it on test would be fitting to the answer."""
    ok = [r for r in records if r["parse_ok"] and r["score"] is not None]
    if not ok:
        return 0.5
    best, best_f1 = 0.5, -1.0
    cands = sorted({round(float(r["score"]), 3) for r in ok} | {0.5})
    for th in cands:
        pred = [float(r["score"]) >= th for r in ok]
        f1 = M.binary_metrics([r["truth"] for r in ok], pred)["f1"]
        if f1 > best_f1:
            best, best_f1 = th, f1
    return best


def evaluate(task_type: str, records: list[dict],
             threshold: float | None = None) -> dict:
    """Pooled + per-user metrics for one condition.

    `threshold` re-derives the decision from help_probability. Models are
    heavily biased on this task - deepseek-v4-flash emitted probabilities in
    0.0-0.7 while the implicit cut sits at 0.5, giving recall ~0. AUPRC/AUROC
    are threshold-free and stay the primary numbers; F1 is reported at a
    threshold calibrated on validation.
    """
    ok = [r for r in records if r["parse_ok"]]
    if threshold is not None:
        for r in ok:
            if r.get("score") is not None:
                r["pred"] = float(r["score"]) >= threshold
    base = {"n_records": len(records), "n_parsed": len(ok),
            "parse_rate": round(len(ok) / len(records), 4) if records else 0.0}
    if not ok:
        return base

    if task_type in ("timing_annotated", "timing_production"):
        base.update(M.binary_metrics([r["truth"] for r in ok],
                                     [r["pred"] for r in ok],
                                     [r["score"] for r in ok]))
        per_user = {}
        by_u = collections.defaultdict(list)
        for r in ok:
            by_u[r["user_id"]].append(r)
        for u, rs in by_u.items():
            per_user[u] = M.binary_metrics([r["truth"] for r in rs],
                                           [r["pred"] for r in rs])
        base["per_user"] = per_user
        base.update(M.user_macro(per_user, "f1"))
        base.update(M.user_macro(per_user, "accuracy"))

    elif task_type == "intention":
        base.update(M.multiclass_metrics([r["truth"] for r in ok],
                                         [r["pred"] for r in ok],
                                         agent.INTENTIONS))
        by_u = collections.defaultdict(list)
        for r in ok:
            by_u[r["user_id"]].append(r)
        per_user = {u: M.multiclass_metrics([r["truth"] for r in rs],
                                            [r["pred"] for r in rs],
                                            agent.INTENTIONS)
                    for u, rs in by_u.items()}
        base["per_user"] = per_user
        base.update(M.user_macro(per_user, "macro_f1"))

    elif task_type == "suggestion":
        base["mean_n_suggestions"] = round(
            sum(len(r["pred"]) for r in ok) / len(ok), 2)
        base["distinct_2"] = M.distinct_n([r["pred"] for r in ok])
        base["note"] = ("SemanticPass@k requires an embedding pass; "
                        "run score_suggestions.py on predictions.jsonl")
    return base


def run(args) -> None:
    ts = tasks.build(args.task)
    engine = LLMEngine(model=args.model, dry_run=not args.live,
                       budget_usd=args.budget, temperature=args.temperature,
                       max_tokens=args.max_tokens, cache=not args.no_cache)
    providers = mem.default_providers(ts, include_controls=not args.no_controls,
                                      engine=engine)
    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",")]
        missing = [c for c in want if c not in providers]
        if missing:
            sys.exit(f"unknown conditions {missing}; available: {sorted(providers)}")
        providers = {k: providers[k] for k in want}

    instances = select_instances(ts, args.split, args.limit, args.min_history)
    if args.users or args.n_users:
        # --limit alone stratifies round-robin over every user, which gives a
        # few instances each across all of them - the opposite of a pilot on a
        # few users with deep history. Restrict the user set first.
        if args.users:
            keep = {u.strip() for u in args.users.split(",")}
            unknown = keep - {i.user_id for i in instances}
            if unknown:
                sys.exit(f"unknown users {sorted(unknown)}")
        else:
            counts = collections.Counter(i.user_id for i in instances)
            keep = {u for u, _ in counts.most_common(args.n_users)}
        instances = [i for i in instances if i.user_id in keep]
    if not instances:
        sys.exit("no instances matched; try --min-history 0 or --split all")

    react = ReactAgent(engine, max_steps=args.max_steps) if args.mode == "react" else None

    run_id = args.run_id or f"{args.task}_{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    print(f"run       : {run_id}")
    print(f"task      : {args.task}  ({len(instances)} instances, "
          f"{len({i.user_id for i in instances})} users, split={args.split})")
    print(f"conditions: {list(providers)}")
    print(f"model     : {args.model}  agent={args.mode}  "
          f"mode={'LIVE' if args.live else 'DRY-RUN'}  workers={args.workers}")
    print()

    results: dict[str, list[dict]] = {}
    t_start = time.time()

    def one(cond_name, provider, inst):
        """Evaluate a single (condition, instance). Safe to run concurrently:
        the engine locks its counters and cache, and providers are read-only."""
        history = ts.history_of(inst)
        card = provider.build(inst, history)
        if react is not None:
            pred, trace = react.run(inst, history, provider)
            raw_text = pred.raw
            messages = [{"role": "meta", "content": f"react:{cond_name}"}]
            tok = (trace.prompt_tokens, trace.completion_tokens)
            latency, cached, err = trace.latency_ms, False, None
        else:
            trace = None
            messages = agent.build_messages(inst, card)
            r = engine.complete(messages, tag=f"{cond_name}:{inst.task_id}")
            pred = agent.parse(r.text, inst.task_type)
            raw_text = r.text
            tok = (r.prompt_tokens, r.completion_tokens)
            latency, cached = r.latency_ms, r.cached
            err = r.error if r.error != "dry_run" else None

        if inst.task_type in ("timing_annotated", "timing_production"):
            pv, score = pred.need_help, pred.help_probability
        elif inst.task_type == "intention":
            pv, score = pred.intention, None
        else:
            pv, score = pred.suggestions, None

        rec = {
            "run_id": run_id, "condition": cond_name,
            "task_id": inst.task_id, "user_id": inst.user_id,
            "t0": inst.t0.isoformat(), "split": inst.split,
            "n_history": len(history),
            "profile_chars": len(card.text), "profile_kind": card.kind,
            # full context needed to recompute metrics or re-parse offline:
            # the model's raw output, the exact profile it saw, and provider
            # metadata (retrieval max_sim, reconstruction bucket/staleness).
            "raw_response": raw_text,
            "profile_text": card.text,
            "profile_meta": card.meta,
            "truth": inst.target["label"], "pred": pv, "score": score,
            "rationale": pred.rationale,
            "parse_ok": pred.parse_ok, "parse_error": pred.parse_error,
            "prompt_hash": _prompt_hash(messages),
            "prompt_tokens": tok[0], "completion_tokens": tok[1],
            "latency_ms": round(latency, 1), "cached": cached,
            "llm_error": err, "mode": args.mode,
        }
        if trace is not None:
            rec["react_steps"] = trace.to_json()
            rec["react_actions"] = trace.actions_used
            rec["react_llm_calls"] = trace.n_llm_calls
            rec["react_stop_reason"] = trace.stop_reason
        if inst.task_type == "suggestion":
            rec["reference_next"] = inst.target["label"]
            rec["deployed_suggestions"] = inst.target.get("deployed_suggestions")
            rec["deployed_suggestions_list"] = inst.target.get(
                "deployed_suggestions_list")
        return rec

    with open(pred_path, "w", encoding="utf-8") as fh:
        for cond_name, provider in providers.items():
            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    records = list(pool.map(
                        lambda i: one(cond_name, provider, i), instances))
            else:
                records = [one(cond_name, provider, i) for i in instances]
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            results[cond_name] = records
            n_ok = sum(r["parse_ok"] for r in records)
            n_err = sum(bool(r["llm_error"]) for r in records)
            print(f"  {cond_name:<18} parsed {n_ok}/{len(records)}  "
                  f"errors {n_err}  ${engine.usage.cost_usd:.3f}  "
                  f"{time.time() - t_start:.0f}s", flush=True)

    thresholds: dict[str, float] = {}
    if args.calibrate_from:
        cal_path = RUNS / args.calibrate_from / "predictions.jsonl"
        if not cal_path.exists():
            sys.exit(f"no predictions at {cal_path}")
        cal = collections.defaultdict(list)
        for line in open(cal_path, encoding="utf-8"):
            rec = json.loads(line)
            cal[rec["condition"]].append(rec)
        for cond, recs in cal.items():
            if recs and recs[0].get("split") == "test":
                print(f"WARNING: calibration source {args.calibrate_from} is a "
                      f"TEST split - thresholds would be fit to the answer")
            thresholds[cond] = pick_threshold(recs)
        print(f"calibrated thresholds: {thresholds}")

    summary = {
        "run_id": run_id,
        "thresholds": thresholds,
        "task": args.task, "model": args.model, "live": bool(args.live),
        "mode": args.mode, "max_steps": args.max_steps,
        "split": args.split, "limit": args.limit,
        "n_instances": len(instances),
        "n_users": len({i.user_id for i in instances}),
        "taskset": ts.describe(),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "usage": engine.usage.summary(),
        "conditions": {c: evaluate(args.task, r, thresholds.get(c))
                       for c, r in results.items()},
    }

    # paired comparisons against every control and against no-memory
    comparisons = {}
    if args.task in ("timing_annotated", "timing_production", "intention"):
        key = "macro_f1" if args.task == "intention" else "f1"
        def per_user_scores(cond):
            pu = summary["conditions"][cond].get("per_user", {})
            return {u: m.get(key) for u, m in pu.items()}
        for cond in results:
            if cond.startswith("ctrl_") or cond == "b0_none":
                continue
            for ref in [c for c in results if c.startswith("ctrl_") or c == "b0_none"]:
                comparisons[f"{cond}__vs__{ref}"] = M.paired_wilcoxon(
                    per_user_scores(cond), per_user_scores(ref))
    summary["paired_comparisons"] = comparisons

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

    if not args.live and engine.dry_run_prompts:
        with open(out_dir / "sample_prompts.txt", "w", encoding="utf-8") as fh:
            for p in engine.dry_run_prompts[:6]:
                fh.write(f"===== {p['tag']} =====\n")
                for m in p["messages"]:
                    fh.write(f"--- {m['role']} ---\n{m['content']}\n\n")

    print()
    print(json.dumps(engine.usage.summary(), indent=2))
    print(f"\nwrote {pred_path}")
    print(f"      {out_dir / 'summary.json'}")
    if not args.live:
        print(f"      {out_dir / 'sample_prompts.txt'}  <- inspect before going live")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="timing_annotated", choices=list(tasks.BUILDERS))
    ap.add_argument("--conditions", help="comma-separated; default = all")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--users", help="comma-separated user ids to restrict to")
    ap.add_argument("--n-users", type=int, default=None,
                    help="restrict to the N users with the most instances")
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--live", action="store_true", help="actually call the API")
    ap.add_argument("--budget", type=float, default=2.0, help="USD hard cap")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="600 truncated reasoning models mid-JSON")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--mode", default="single", choices=["single", "react"],
                    help="single-shot prompt, or ReAct with memory-backed tools")
    ap.add_argument("--max-steps", type=int, default=3,
                    help="max tool calls before the agent must answer")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests; 1 = sequential")
    ap.add_argument("--calibrate-from", metavar="RUN_ID",
                    help="fit the decision threshold per condition on a previous "
                         "val-split run (never on test)")
    ap.add_argument("--run-id")
    ap.add_argument("--healthcheck", action="store_true")
    args = ap.parse_args()

    if args.healthcheck:
        eng = LLMEngine(model=args.model, dry_run=True)
        r = eng.check_connection()
        print(f"model={args.model} ok={r.ok} text={r.text[:60]!r} error={r.error}")
        return
    try:
        run(args)
    except BudgetExceeded as exc:
        sys.exit(f"BUDGET STOP: {exc}")
    except ProviderOutOfCredit as exc:
        sys.exit(f"OUT OF CREDIT: {exc}")


if __name__ == "__main__":
    main()
