"""Task instances for the PIPA experiments.

Everything a condition may see is inside `TaskInstance.observation`. Everything
being predicted is inside `.target`. Nothing else is passed to a model, so a
leak has to be introduced deliberately rather than by accident.

Three task types:

    timing_annotated   E1  need_help at a screen moment      (labelled, 14 users)
    intention          E2  14-class intent, positives only    (labelled, 14 users)
    timing_production  E1b did the user engage within 5 min   (behavioural, 26 users)
    suggestion         E3  what did the user actually do next (outcomes, 17 users)

Chronology is enforced at construction: `history_ids` for an instance only ever
references instances of the same user with a strictly earlier timestamp.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATASET = PROJECT / "pipa_dataset"
sys.path.insert(0, str(DATASET / "scripts"))

TaskType = Literal["timing_annotated", "intention", "timing_production", "suggestion"]

# How many prior events an instance may reference. The earlier value of 200
# discarded 71% of the available record (true median 507, max 2,426) and pinned
# every instance at the cap, which made history-depth, cold-start and profile
# staleness analyses degenerate. 512 covers ~50% of instances fully while
# keeping retrieval and profile construction affordable.
MAX_HISTORY = 512


@dataclass
class TaskInstance:
    task_id: str
    task_type: TaskType
    user_id: str
    t0: datetime
    observation: dict[str, Any]
    target: dict[str, Any]
    # ids of the same user's strictly-earlier instances, oldest first
    history_ids: list[str] = field(default_factory=list)
    split: str = "train"

    def __repr__(self) -> str:  # keep logs readable
        return (f"TaskInstance({self.task_type} {self.task_id} user={self.user_id} "
                f"hist={len(self.history_ids)})")


@dataclass
class TaskSet:
    task_type: TaskType
    instances: list[TaskInstance]

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[TaskInstance]:
        return iter(self.instances)

    @property
    def by_id(self) -> dict[str, TaskInstance]:
        if not hasattr(self, "_by_id"):
            self._by_id = {i.task_id: i for i in self.instances}
        return self._by_id

    def split_of(self, name: str) -> list[TaskInstance]:
        return [i for i in self.instances if i.split == name]

    def users(self) -> list[str]:
        return sorted({i.user_id for i in self.instances})

    def history_of(self, inst: TaskInstance) -> list[TaskInstance]:
        """Resolve history ids to instances. Guaranteed strictly before t0."""
        lookup = self.by_id
        out = [lookup[h] for h in inst.history_ids if h in lookup]
        assert all(h.t0 < inst.t0 for h in out), "history chronology violated"
        return out

    def describe(self) -> dict:
        pos = [i for i in self.instances if i.target.get("label") is True]
        return {
            "task_type": self.task_type,
            "instances": len(self.instances),
            "users": len(self.users()),
            "splits": pd.Series([i.split for i in self.instances]).value_counts().to_dict(),
            "positive_rate": round(len(pos) / len(self.instances), 4) if self.instances else 0,
            "median_history": int(np.median([len(i.history_ids) for i in self.instances]))
            if self.instances else 0,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _attach_history(instances: list[TaskInstance], max_history: int = MAX_HISTORY) -> None:
    """Link each instance to the same user's strictly earlier instances."""
    instances.sort(key=lambda i: (i.user_id, i.t0))
    seen: dict[str, list[TaskInstance]] = {}
    for inst in instances:
        prior = seen.setdefault(inst.user_id, [])
        # strict inequality: simultaneous events are not usable as history
        inst.history_ids = [p.task_id for p in prior if p.t0 < inst.t0][-max_history:]
        prior.append(inst)


# --------------------------------------------------------------------------
# annotated tasks
# --------------------------------------------------------------------------


def _load_annotated():
    from load import load_annotated  # noqa: E402
    return load_annotated()


def build_timing_annotated(max_history: int = MAX_HISTORY) -> TaskSet:
    ds = _load_annotated()
    splits = ds.splits["temporal"]
    split_of = {sid: name for name in ("train", "val", "test") for sid in splits[name]}

    hist = ds.history.sort_values(["sample_id", "start_time"])
    hist_by_sample = {sid: g for sid, g in hist.groupby("sample_id")}

    instances: list[TaskInstance] = []
    for row in ds.samples.itertuples():
        recent = hist_by_sample.get(row.sample_id)
        recent_events = []
        if recent is not None:
            for h in recent.tail(8).itertuples():
                recent_events.append({
                    "seconds_before": round(float(h.seconds_before_t0), 1),
                    "app": h.app_name,
                    "window": h.window_title,
                    "summary": h.summary,
                })
        instances.append(TaskInstance(
            task_id=row.sample_id,
            task_type="timing_annotated",
            user_id=row.user_id,
            t0=pd.Timestamp(row.t0).to_pydatetime(),
            observation={
                "app_name": row.app_name,
                "window_title": row.window_title,
                "screen_summary": row.context_summary,
                "recent_activity": recent_events,
                "hour_of_day": int(row.hour_of_day),
                "weekday": int(row.weekday),
            },
            target={"label": bool(row.need_help),
                    "intention": row.intention if row.need_help else None},
            split=split_of.get(row.sample_id, "train"),
        ))
    _attach_history(instances, max_history)
    return TaskSet("timing_annotated", instances)


def build_intention(max_history: int = MAX_HISTORY) -> TaskSet:
    """Positives only - intention is a conditional label (see DATASET.md 2.5)."""
    base = build_timing_annotated(max_history)
    keep = [i for i in base.instances if i.target["label"] and i.target["intention"]]
    kept_ids = {i.task_id for i in keep}
    out = []
    for inst in keep:
        out.append(TaskInstance(
            task_id=inst.task_id,
            task_type="intention",
            user_id=inst.user_id,
            t0=inst.t0,
            observation=inst.observation,
            target={"label": inst.target["intention"]},
            history_ids=[h for h in inst.history_ids if h in kept_ids],
            split=inst.split,
        ))
    return TaskSet("intention", out)


# --------------------------------------------------------------------------
# production tasks
# --------------------------------------------------------------------------


def _load_texts() -> pd.DataFrame:
    t = pd.read_parquet(DATASET / "production" / "texts_gateway.parquet")
    t["ts"] = pd.to_datetime(t["timestamp"], errors="coerce")
    # 6.6% of foreground rows carry no user attribution - unusable per-user.
    return t[(t.user_id != "") & t.ts.notna()]


# System-injected turns are not user utterances. These markers appear in
# prompts the product pushes into the chat (SOP blocks, skill scaffolding).
_INJECTED_MARKERS = (
    "=== ", "禁止 get_skill_content", "SOP 已注入", "阶段 0", "入口：",
    "【当前选中待办】",
)


def _is_real_utterance(text: str) -> bool:
    if not text or len(text.strip()) < 4:
        return False
    return not any(m in text for m in _INJECTED_MARKERS)


def _clean_user_turn(text: str) -> str:
    """Strip the product's '用户输入:' wrapper to get the raw utterance."""
    for marker in ("用户输入:", "用户输入："):
        if marker in text:
            text = text.split(marker, 1)[1]
    return text.strip()


def build_timing_production(window_seconds: int = 300,
                            max_history: int = MAX_HISTORY,
                            min_gap_seconds: int | None = None) -> TaskSet:
    """Screen caption -> did a real chat follow within `window_seconds`?

    Positives and negatives come from the same caption stream, so unlike the
    annotated set there is no asymmetric sampling to leak the label.
    """
    t = _load_texts()
    caps = t[t.family == "screen_caption"].sort_values("ts")
    chats = t[t.family == "foreground_chat"].sort_values("ts")
    users = sorted(set(caps.user_id) & set(chats.user_id))

    instances: list[TaskInstance] = []
    for user in users:
        c = caps[caps.user_id == user]
        ch = chats[chats.user_id == user]
        chat_ts = ch.ts.to_numpy()
        for r in c.itertuples():
            idx = int(np.searchsorted(chat_ts, np.datetime64(r.ts)))
            engaged = False
            if idx < len(chat_ts):
                gap = (chat_ts[idx] - np.datetime64(r.ts)) / np.timedelta64(1, "s")
                engaged = 0 < gap <= window_seconds
            instances.append(TaskInstance(
                task_id=f"pt_{r.request_id}",
                task_type="timing_production",
                user_id=user,
                t0=r.ts.to_pydatetime(),
                observation={
                    "screen_summary": (r.assistant_text or "")[:2000],
                    "hour_of_day": r.ts.hour,
                    "weekday": r.ts.weekday(),
                },
                target={"label": bool(engaged)},
            ))
    _attach_history(instances, max_history)

    # Captions arrive ~44 s apart while each label looks ahead `window_seconds`
    # (300 s), so 83% of consecutive instances had OVERLAPPING label windows and
    # a single chat event marked a dozen captions positive. That made the label
    # autocorrelated - P(y=1 | previous y=1) = 0.810 against a 0.123 base rate -
    # and "repeat the previous label" scored 0.670 AUPRC, more than double any
    # method. Thinning to non-overlapping windows removes the artefact.
    #
    # As with build_suggestion, thinned instances stay in HISTORY and are only
    # excluded from EVALUATION, so memory keeps the full record.
    gap = window_seconds if min_gap_seconds is None else min_gap_seconds
    if gap:
        by_user: dict[str, list[TaskInstance]] = {}
        for inst in instances:
            by_user.setdefault(inst.user_id, []).append(inst)
        keep: set[int] = set()
        for group in by_user.values():
            group.sort(key=lambda i: i.t0)
            last = None
            for inst in group:
                if last is None or (inst.t0 - last).total_seconds() >= gap:
                    keep.add(id(inst))
                    last = inst.t0
        evaluable = [i for i in instances if id(i) in keep]
        for i in instances:
            if id(i) not in keep:
                i.split = "duplicate"
        _assign_temporal_splits(evaluable)
        return TaskSet("timing_production", instances)

    _assign_temporal_splits(instances)
    return TaskSet("timing_production", instances)


def build_suggestion(window_seconds: int = 3600,
                     max_history: int = MAX_HISTORY,
                     dedupe_targets: bool = True) -> TaskSet:
    """Agent emitted suggestions -> what the user actually did next.

    The only task in the project with real-world ground truth.
    """
    t = _load_texts()
    sug = t[(t.family == "followup_suggestions")
            & (t.assistant_text.str.len() > 0)].sort_values("ts")
    chats = t[t.family == "foreground_chat"].copy()
    chats["utterance"] = chats.user_text.map(_clean_user_turn)
    chats = chats[chats.utterance.map(_is_real_utterance)]
    # one user turn fans out into several ReAct calls with identical text
    chats = chats.drop_duplicates(["user_id", "utterance"]).sort_values("ts")

    instances: list[TaskInstance] = []
    for user in sorted(set(sug.user_id) & set(chats.user_id)):
        s = sug[sug.user_id == user]
        c = chats[chats.user_id == user]
        c_ts = c.ts.to_numpy()
        for r in s.itertuples():
            idx = int(np.searchsorted(c_ts, np.datetime64(r.ts)))
            if idx >= len(c_ts):
                continue
            gap = (c_ts[idx] - np.datetime64(r.ts)) / np.timedelta64(1, "s")
            if not (0 < gap <= window_seconds):
                continue
            nxt = c.iloc[idx]
            # If the "next" utterance already appears in the suggestion's own
            # input, it is not a future event - it is the same turn echoed back
            # (menu selections, project-context blocks). Scoring against it
            # would be circular.
            utt = str(nxt.utterance).strip()
            if len(utt) >= 12 and utt[:40] in (r.user_text or ""):
                continue
            instances.append(TaskInstance(
                task_id=f"sg_{r.request_id}",
                task_type="suggestion",
                user_id=user,
                t0=r.ts.to_pydatetime(),
                observation={
                    "context": r.user_text[:3000],
                    "hour_of_day": r.ts.hour,
                    "weekday": r.ts.weekday(),
                },
                target={
                    "label": nxt.utterance,
                    "seconds_to_next": round(float(gap), 1),
                    "deployed_suggestions": r.assistant_text[:2000],
                    # parsed list for the B6 baseline; empty means the deployed
                    # output was malformed and this instance should be skipped
                    # when comparing against it (~6% of pairs)
                    "deployed_suggestions_list":
                        parse_deployed_suggestions(r.assistant_text),
                },
            ))
    _attach_history(instances, max_history)

    if dedupe_targets:
        # Several followup_suggestions events precede the SAME next user turn,
        # so the identical target repeats and "echo the last request" scores
        # 0.796 pass@3 by reading the answer off its own input.
        #
        # Duplicates are excluded from EVALUATION but kept in HISTORY: the fact
        # that a user asks something repeatedly is real personalisation signal,
        # and removing it from memory would destroy the frequency information
        # the repertoire depends on. Marking them with a non-standard split
        # keeps them reachable as history while --split test/val/train never
        # selects them.
        # Keep the FIRST occurrence of each distinct request, not the last.
        # Keeping the last leaves earlier duplicates in history carrying the
        # exact target label - 35.8% of targets were recoverable that way. At
        # the first occurrence the user has not asked it before, so the task is
        # genuinely "predict a request they have not yet made".
        first: dict[tuple[str, str], TaskInstance] = {}
        for inst in instances:
            key = (inst.user_id, str(inst.target.get("label") or "").strip())
            cur = first.get(key)
            if cur is None or inst.t0 < cur.t0:
                first[key] = inst
        keep = {id(i) for i in first.values()}
        # A target can also appear inside a PAST INSTANCE'S CONTEXT without ever
        # having been a past label (the user said it, but the next turn was
        # something else). Conditions that render raw context would then expose
        # the answer. 4 of 254 instances (1.6%); dropped.
        by_user_all: dict[str, list[TaskInstance]] = {}
        for inst in instances:
            by_user_all.setdefault(inst.user_id, []).append(inst)
        for group in by_user_all.values():
            group.sort(key=lambda i: i.t0)
            for pos, inst in enumerate(group):
                if id(inst) not in keep:
                    continue
                tgt = str(inst.target.get("label") or "").strip()
                if len(tgt) < 12:
                    continue
                past_ctx = " ".join(
                    (g.observation.get("context") or "") for g in group[:pos])
                if tgt[:40] in past_ctx:
                    keep.discard(id(inst))
        evaluable = [i for i in instances if id(i) in keep]
        for i in instances:
            if id(i) not in keep:
                i.split = "duplicate"
        _assign_temporal_splits(evaluable)
        return TaskSet("suggestion", instances)

    _assign_temporal_splits(instances)
    return TaskSet("suggestion", instances)


def parse_deployed_suggestions(raw: str) -> list[str]:
    """Extract the product's own suggestions for the B6 baseline comparison.

    Not all rows are usable: of 1,046 pairs, 981 are proper suggestion JSON,
    38 are numeric arrays, 25 are empty lists and 2 use a `follow_ups` key.
    Callers should skip instances where this returns [] rather than scoring
    the deployed baseline against garbage. The task's ground truth (the user's
    actual next utterance) is unaffected.
    """
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except ValueError:
        return []
    if isinstance(obj, dict):
        for key in ("suggestions", "follow_ups", "followups"):
            v = obj.get(key)
            if isinstance(v, list):
                return [str(s).strip() for s in v if isinstance(s, str) and s.strip()]
        return []
    if isinstance(obj, list):
        return [s.strip() for s in obj if isinstance(s, str) and s.strip()]
    return []


def _assign_temporal_splits(instances: list[TaskInstance],
                            fracs=(0.70, 0.15)) -> None:
    """Per-user chronological split, matching the annotated protocol."""
    by_user: dict[str, list[TaskInstance]] = {}
    for i in instances:
        by_user.setdefault(i.user_id, []).append(i)
    for group in by_user.values():
        group.sort(key=lambda i: i.t0)
        n = len(group)
        a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
        for k, inst in enumerate(group):
            inst.split = "train" if k < a else ("val" if k < b else "test")


BUILDERS = {
    "timing_annotated": build_timing_annotated,
    "intention": build_intention,
    "timing_production": build_timing_production,
    "suggestion": build_suggestion,
}


def build(task_type: TaskType, **kw) -> TaskSet:
    return BUILDERS[task_type](**kw)


if __name__ == "__main__":
    for name in BUILDERS:
        ts = build(name)
        print(json.dumps(ts.describe(), ensure_ascii=False))
