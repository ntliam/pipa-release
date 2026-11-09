"""Generate a small synthetic corpus so the pipeline runs without real data.

    python pipa_dataset/scripts/make_synthetic.py --users 12 --per-user 60

The corpora this project was developed on are logs of real people's screens
and chats and are not redistributable. This script writes files with the same
schema and the same loader contract, so every entry point in `pipa/` runs
end to end against it.

What it is for: verifying the code runs, reading the data flow, and developing
against a stable fixture. What it is NOT for: reproducing any number in the
paper. The users here are drawn from a Beta prior with no real structure, so a
personalised model will beat a population baseline (receptivity genuinely
varies by user) but the absolute values mean nothing.

Schema (see pipa_dataset/scripts/load.py for the loader that consumes it):

  annotated/samples.parquet          one row per decision moment
  annotated/history_events.parquet   the events preceding each moment
  annotated/context_screens.parquet  screen captures for each moment
  annotated/splits.json              per-user chronological 70/15/15
  production/texts_gateway.parquet   gateway log: screen_caption + foreground_chat
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APPS = ["Code", "Chrome", "Slack", "Terminal", "Figma", "Notion", "Mail", "Excel"]
INTENTIONS = ["knowledge_qa", "data_analysis", "advice", "lookup",
              "coding", "generation", "summarise", "translate"]
SCREEN = {
    "Code": "editing {f}.py, {n} lines, a failing test in the panel below",
    "Chrome": "reading documentation for {f}, several tabs open",
    "Slack": "a thread in #{f} with {n} unread replies",
    "Terminal": "running a build in {f}/, {n} warnings so far",
    "Figma": "a layout called {f}, inspecting a component",
    "Notion": "a page titled '{f}' with a half-written checklist",
    "Mail": "a draft reply to a message about {f}",
    "Excel": "a sheet '{f}' with a pivot over {n} rows",
}
TOPIC = ["billing", "onboarding", "retrieval", "latency", "pricing", "schema",
         "auth", "rollout", "metrics", "caching"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--per-user", type=int, default=90,
                    help="decision moments per user; >=80 so the default --min-history=64 "
                         "in profile_unsupervised.py is satisfied")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Users differ in how often they want help; that is the thing a personalised
    # model has to recover, so it must actually vary.
    theta = rng.beta(2.0, 2.0, size=args.users)
    t_start = datetime(2026, 1, 6, 9, 0, 0)

    samples, history, screens, per_user_ids = [], [], [], {}
    for u in range(args.users):
        uid = f"u_synth{u:03d}"
        per_user_ids[uid] = []
        t = t_start + timedelta(hours=float(rng.integers(0, 72)))
        for k in range(args.per_user):
            t += timedelta(minutes=float(rng.integers(7, 90)))
            sid = f"s_{uid}_{k:04d}"
            per_user_ids[uid].append(sid)
            app = APPS[int(rng.integers(len(APPS)))]
            topic = TOPIC[int(rng.integers(len(TOPIC)))]
            summary = SCREEN[app].format(f=topic, n=int(rng.integers(3, 400)))
            need = bool(rng.random() < theta[u])
            samples.append({
                "sample_id": sid, "user_id": uid, "user_name": f"synthetic user {u}",
                "t0": t, "hour_of_day": t.hour, "weekday": t.weekday(),
                "app_name": app, "window_title": f"{topic} - {app}",
                "n_context_screens": 1, "context_summary": summary,
                "n_history_events": 8, "history_span_seconds": 1800,
                "n_history_apps": int(rng.integers(1, 5)), "history_app_switches": int(rng.integers(0, 6)),
                "need_help": need,
                "intention": INTENTIONS[int(rng.integers(len(INTENTIONS)))] if need else None,
            })
            screens.append({"sample_id": sid, "screen_order": 0, "t": t,
                            "app_name": app, "window_title": f"{topic} - {app}",
                            "summary": summary})
            for j in range(8):          # the 8 events before the moment
                dt = int(rng.integers(30, 1800))
                ha = APPS[int(rng.integers(len(APPS)))]
                ht = TOPIC[int(rng.integers(len(TOPIC)))]
                history.append({
                    "sample_id": sid, "start_time": t - timedelta(seconds=dt),
                    "seconds_before_t0": float(dt), "app_name": ha,
                    "window_title": f"{ht} - {ha}",
                    "summary": SCREEN[ha].format(f=ht, n=int(rng.integers(3, 400))),
                })

    # per-user chronological 70/15/15, matching the real protocol
    splits = {"temporal": {"train": [], "val": [], "test": []},
              "user_holdout": {"train_users": [], "val_users": [], "test_users": []}}
    for uid, sids in per_user_ids.items():
        n = len(sids); a, b = int(n * 0.70), int(n * 0.85)
        splits["temporal"]["train"] += sids[:a]
        splits["temporal"]["val"] += sids[a:b]
        splits["temporal"]["test"] += sids[b:]
    us = list(per_user_ids)
    a, b = int(len(us) * 0.7), int(len(us) * 0.85)
    splits["user_holdout"]["train_users"] = us[:a]
    splits["user_holdout"]["val_users"] = us[a:b]
    splits["user_holdout"]["test_users"] = us[b:]

    ann = ROOT / "annotated"; ann.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_parquet(ann / "samples.parquet", index=False)
    pd.DataFrame(history).to_parquet(ann / "history_events.parquet", index=False)
    pd.DataFrame(screens).to_parquet(ann / "context_screens.parquet", index=False)
    (ann / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    # production: a gateway log of screen captions, with a chat sometimes
    # following within five minutes - that proximity IS the behavioural label
    rows = []
    for u in range(args.users):
        uid = f"u_synth{u:03d}"
        t = t_start + timedelta(hours=float(rng.integers(0, 72)))
        for k in range(args.per_user * 2):
            t += timedelta(seconds=float(rng.integers(40, 400)))
            app = APPS[int(rng.integers(len(APPS)))]
            topic = TOPIC[int(rng.integers(len(TOPIC)))]
            rows.append({"request_id": f"r_{uid}_{k:05d}", "user_id": uid,
                         "timestamp": t, "family": "screen_caption",
                         "is_foreground": True, "system_prompt_id": "caption_v1",
                         "user_text": None,
                         "assistant_text": SCREEN[app].format(f=topic, n=int(rng.integers(3, 400)))})
            if rng.random() < theta[u] * 0.5:
                rows.append({"request_id": f"c_{uid}_{k:05d}", "user_id": uid,
                             "timestamp": t + timedelta(seconds=float(rng.integers(5, 290))),
                             "family": "foreground_chat", "is_foreground": True,
                             "system_prompt_id": "chat_v1",
                             "user_text": f"can you help me with the {topic} thing in {app}?",
                             "assistant_text": "Sure - here is what I would do."})
    prod = ROOT / "production"; prod.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(prod / "texts_gateway.parquet", index=False)

    print(f"synthetic corpus written to {ROOT}")
    print(f"  annotated : {len(samples):,} samples, {len(history):,} history events, "
          f"{args.users} users")
    print(f"  production: {len(rows):,} gateway rows "
          f"({sum(r['family'] == 'foreground_chat' for r in rows):,} chats)")
    print(f"  true receptivity ranges {theta.min():.2f}-{theta.max():.2f} across users")
    print("\nSynthetic data. Use it to check the code runs, not to reproduce results.")


if __name__ == "__main__":
    main()
