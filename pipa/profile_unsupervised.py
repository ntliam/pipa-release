"""Unsupervised evaluation of profile reconstruction.

    python profile_unsupervised.py --task timing_annotated --live --budget 4

The internal corpora have no hand-written user model, so accuracy cannot be
scored the way it was against Pi-Bench's `profile.yaml`. But reconstruction
does not need annotation to be evaluated: **the user's later behaviour is the
label.** A profile built from the first half of a history either predicts the
second half or it does not, and nothing has to be labelled for that to be
checked.

Three criteria, none requiring ground truth:

  U1  held-out rate       the profile states how often this person wants help.
                          Score it against the SECOND half only. The existing
                          Discovery Accuracy compares against the whole corpus,
                          which includes the events the profile was built from -
                          a mild leak that flatters it.

  U2  contrastive recall  given the profile and k candidate events - one really
                          this user's, k-1 drawn from other users - can the
                          profile pick the real one? Chance is 1/k, every logged
                          event is an instance, and no annotation exists to be
                          wrong. Run with the user's own profile and with a
                          donor's: identity, not fluency, has to carry it.

  U3  discrimination gap  U2 with own profile minus U2 with a donor profile.
                          This is the number that matters: a profile that helps
                          equally whoever it belongs to is a genre template.

U2 is the criterion worth keeping. It turns an unlabelled corpus into an
evaluation set of arbitrary size, which is the standing limitation of this
project - 14 to 26 users and a few hundred labelled instances.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Force UTF-8 on stdout. reconfigure() is idempotent; re-wrapping with
# TextIOWrapper closes the shared buffer when a second module does the
# same in one process, which breaks any script that imports two of these.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent

RATE_SYS = ("You read a description of a person and answer with a single "
            "percentage. No words, no explanation.")
RATE_TMPL = """PROFILE OF A PERSON

{profile}

Out of the next 100 moments observed for this person, in how many will they
want an assistant to step in and help? Answer with a number 0-100 only."""

PICK_SYS = (
    "You are given a description of a person and several observed moments. "
    "Exactly one moment is from that person's own record; the others belong to "
    "different people. Identify the one that belongs to the described person.\n\n"
    "Answer with the number only."
)
PICK_TMPL = """PROFILE OF A PERSON

{profile}

CANDIDATE MOMENTS

{candidates}

Which numbered moment is from this person's record? Number only."""

_NUM = re.compile(r"\d+")


def _first_int(text: str, lo: int, hi: int):
    for m in _NUM.finditer(text or ""):
        v = int(m.group())
        if lo <= v <= hi:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="timing_annotated")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--min-history", type=int, default=64)
    ap.add_argument("--max-users", type=int, default=10)
    ap.add_argument("--trials", type=int, default=12, help="U2 trials per user")
    ap.add_argument("-k", type=int, default=5, help="candidates per U2 trial")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    import tasks
    from engine import LLMEngine
    from profile import ProfileReconstructor
    from memory import _describe

    ts = tasks.build(args.task)
    eng = LLMEngine(model=args.model, dry_run=not args.live,
                    budget_usd=args.budget, temperature=0.0, max_tokens=400)
    recon = ProfileReconstructor(eng)
    rng = random.Random(args.seed)

    per_user: dict[str, list] = {}
    for inst in ts.instances:
        h = ts.history_of(inst)
        if len(h) > len(per_user.get(inst.user_id, [])):
            per_user[inst.user_id] = h
    users = sorted([u for u, h in per_user.items() if len(h) >= args.min_history],
                   key=lambda u: -len(per_user[u]))[: args.max_users]
    if len(users) < 3:
        raise SystemExit("need >= 3 users")
    print(f"{args.task}: {len(users)} users\n")

    # first half builds the profile; second half is held out entirely
    prof, held = {}, {}
    for u in users:
        h = per_user[u]
        mid = len(h) // 2
        prof[u] = recon._build_prefix(f"{u}#u1", h[:mid]).text
        held[u] = h[mid:]

    def ask(system, user, tag, hi):
        r = eng.complete([{"role": "system", "content": system},
                          {"role": "user", "content": user}], tag=tag)
        return _first_int(r.text, 0, hi) if r.ok else None

    # ---- U1 held-out rate -------------------------------------------------
    print("## U1  Held-out help-rate\n")
    print(f"{'user':14}{'stated':>9}{'actual':>9}{'abs err':>9}")
    errs, u1_rows = [], []
    for u in users:
        labs = [bool(x.target.get("label")) for x in held[u]
                if isinstance(x.target.get("label"), bool)]
        if not labs:
            continue
        actual = sum(labs) / len(labs)
        stated = ask(RATE_SYS, RATE_TMPL.format(profile=prof[u][:6000]),
                     f"u1:{u}", 100)
        if stated is None:
            continue
        e = abs(stated / 100 - actual)
        errs.append(e)
        u1_rows.append({"user": u, "stated": round(stated / 100, 4),
                        "actual": round(actual, 4), "abs_err": round(e, 4),
                        "heldout_events": len(labs)})
        print(f"{u:14}{stated/100:>9.3f}{actual:>9.3f}{e:>9.3f}")
    if errs:
        print(f"{'MAE':14}{'':>9}{'':>9}{sum(errs)/len(errs):>9.3f}")

    # ---- U2/U3 contrastive ------------------------------------------------
    print("\n## U2  Contrastive recall — pick this user's real moment\n")
    res = {"own": [], "donor": []}
    per_user = {}
    for u in users:
        others = [o for o in users if o != u]
        for _ in range(args.trials):
            real = rng.choice(held[u])
            distract = [rng.choice(held[rng.choice(others)])
                        for _ in range(args.k - 1)]
            pool = [real] + distract
            order = list(range(args.k))
            rng.shuffle(order)
            shown = [pool[i] for i in order]
            gold = order.index(0) + 1
            listing = "\n".join(f"{i+1}. {_describe(x)[:220]}"
                                for i, x in enumerate(shown))
            for arm, who in (("own", u), ("donor", rng.choice(others))):
                pick = ask(PICK_SYS,
                           PICK_TMPL.format(profile=prof[who][:6000],
                                            candidates=listing),
                           f"u2:{arm}:{u}", args.k)
                if pick is not None:
                    res[arm].append(int(pick == gold))
                    per_user.setdefault(u, {"own": [], "donor": []})[arm].append(
                        int(pick == gold))
    for arm in ("own", "donor"):
        v = res[arm]
        if v:
            acc = sum(v) / len(v)
            se = (acc * (1 - acc) / len(v)) ** 0.5
            print(f"{arm:8} accuracy {acc:.3f} +/- {se:.3f}  "
                  f"(n={len(v)}, chance {1/args.k:.3f})")
    if res["own"] and res["donor"]:
        a = sum(res["own"]) / len(res["own"])
        b = sum(res["donor"]) / len(res["donor"])
        print(f"\n## U3  Discrimination gap = {a-b:+.3f}  "
              f"(own {a:.3f} vs donor {b:.3f})")

    out = ROOT / f"exports/profile_unsupervised_{args.task}.json"
    out.write_text(json.dumps(
        {"task": args.task, "users": users, "u1_mae": (sum(errs)/len(errs)) if errs else None,
         "u2": {k: (sum(v)/len(v) if v else None) for k, v in res.items()},
         "n_trials": {k: len(v) for k, v in res.items()}, "k": args.k,
         "u1_rows": u1_rows,
         "u2_per_user": {u: {a: {"acc": (sum(v) / len(v)) if v else None,
                                 "n": len(v)}
                             for a, v in arms.items()}
                         for u, arms in per_user.items()}},
        indent=2), encoding="utf-8")
    print(f"\nspend ${eng.usage.cost_usd:.3f}  calls {eng.usage.calls}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
