"""Generative identity test: can PIPA regenerate the user from their profile?

    python run_simulation.py --live --budget 3

The question. Every earlier identity test asked whether the right profile helps
a *decision*. This asks something stronger and more direct: is the profile a
sufficient description of the person? Give the model nothing but the profile -
no raw history - and have it write the activities that person would plausibly
produce. Then check whether the simulation lands nearest that user's REAL
held-out behaviour, among all users.

  profile(u)  ->  simulated activities  ->  centroid s_u
  real held-out activities of u          ->  centroid r_u
  metric: is r_u the nearest real centroid to s_u?   chance = 1/n

Protocol, fixed before looking at any result:

* Chronological split per user: the profile is built from the FIRST 70% of
  their events, the comparison uses the LAST 30%. The simulation therefore
  cannot be memorising the text it is scored against.
* The metric is computed in the FULL embedding space. Any 2-D map is drawn
  afterwards for illustration only - t-SNE/UMAP distances are not faithful and
  must not carry the claim.
* The same backbone writes the profile and the simulation, so shared house
  style is a real risk. It cancels: we do not ask "is this plausible?" but "is
  this nearest the RIGHT user?", and style common to all simulations shifts
  every centroid alike.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import memory as mem
import tasks
from engine import LLMEngine
from profile import ProfileReconstructor

# Force UTF-8 on stdout. reconfigure() is idempotent; re-wrapping with
# TextIOWrapper closes the shared buffer when a second module does the
# same in one process, which breaks any script that imports two of these.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parent / "runs"

SIM_SYSTEM = (
    "You are given a model of one person's working behaviour. Write the kinds "
    "of on-screen activity this specific person would plausibly produce next.\n\n"
    "Write one activity per line, in the same terse style as an activity log: "
    "the application, what is on screen, and what they are doing. Be concrete "
    "and specific to THIS person - their tools, their subject matter, their "
    "habits. Do not write generic office activity that would fit anyone. Do "
    "not number the lines, do not add commentary."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="timing_production")
    ap.add_argument("--min-instances", type=int, default=60)
    ap.add_argument("--n-sim", type=int, default=25, help="activities to generate")
    ap.add_argument("--holdout", type=float, default=0.30)
    ap.add_argument("--max-real", type=int, default=60, help="real events embedded")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    ts = tasks.build(args.task)
    by_user = collections.defaultdict(list)
    for i in ts.instances:
        by_user[i.user_id].append(i)
    for u in by_user:
        by_user[u].sort(key=lambda x: x.t0)
    users = sorted(u for u, v in by_user.items() if len(v) >= args.min_instances)
    print(f"{len(users)} users with >= {args.min_instances} instances "
          f"(chance top-1 = {1/len(users):.3f})")

    engine = LLMEngine(model=args.model, dry_run=not args.live,
                       budget_usd=args.budget, temperature=0.0, max_tokens=1200)
    recon = ProfileReconstructor(engine)

    out = []
    for u in users:
        ev = by_user[u]
        cut = int(len(ev) * (1 - args.holdout))
        early, late = ev[:cut], ev[cut:]
        if not early or not late:
            continue
        card = recon.build(early[-1], early)
        prompt = (f"USER MODEL\n{card.text}\n\n"
                  f"Write {args.n_sim} activities this person would plausibly "
                  f"produce next, one per line.")
        resp = engine.complete(
            [{"role": "system", "content": SIM_SYSTEM},
             {"role": "user", "content": prompt}], tag=f"sim:{u}")
        sims = [l.strip(" -•\t") for l in (resp.text or "").splitlines()
                if len(l.strip(" -•\t")) > 12][: args.n_sim]
        # real held-out behaviour, rendered exactly as the profile's input was
        real = [mem._describe(h) for h in late[: args.max_real]]
        out.append({"user_id": u, "n_early": len(early), "n_late": len(late),
                    "profile_chars": len(card.text), "profile_text": card.text,
                    "simulated": sims, "real_holdout": real})
        print(f"  {u[:14]:<14} profile {len(card.text):>5}ch  "
              f"sim {len(sims):>3}  real {len(real):>3}")

    run_id = args.run_id or f"sim_{datetime.now():%Y%m%d_%H%M%S}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "simulations.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {d}")
    print(json.dumps(engine.usage.summary(), indent=2))
    print(f"\nnext: python score_simulation.py --run {run_id}")


if __name__ == "__main__":
    main()
