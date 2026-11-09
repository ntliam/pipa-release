"""Prompt construction and output parsing.

One prompt builder per task, shared by every condition. The ONLY thing that
varies across B0-B7 is the profile block, so a measured difference is
attributable to the memory representation and nothing else.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from memory import ProfileCard
from tasks import TaskInstance

INTENTIONS = [
    "知识问答", "数据分析", "总结归纳", "对比评测", "信息查询", "代码编程",
    "数学计算", "内容生成", "建议咨询", "文档编辑", "创意设计", "格式转换",
    "翻译", "未分类",
]

SYSTEM = (
    "You are a proactive personal assistant deciding whether and how to help a "
    "user, based only on what is observable right now plus what you know about "
    "this specific user. Answer with valid JSON only - no prose, no markdown "
    "fences."
)


@dataclass
class Prediction:
    raw: str = ""
    need_help: bool | None = None
    help_probability: float | None = None
    intention: str | None = None
    suggestions: list[str] = field(default_factory=list)
    rationale: str = ""
    parse_ok: bool = False
    parse_error: str | None = None


# --------------------------------------------------------------------------
# prompt building
# --------------------------------------------------------------------------


def _observation_block(inst: TaskInstance) -> str:
    o = inst.observation
    lines = [f"Time: {inst.t0:%Y-%m-%d %H:%M} (weekday {o.get('weekday')}, "
             f"hour {o.get('hour_of_day')})"]
    if o.get("app_name"):
        lines.append(f"Active application: {o['app_name']}")
    if o.get("window_title"):
        lines.append(f"Window title: {o['window_title'][:200]}")
    if o.get("screen_summary"):
        lines.append(f"Screen content: {o['screen_summary'][:1200]}")
    if o.get("context"):
        lines.append(f"Conversation context: {o['context'][:1500]}")
    recent = o.get("recent_activity") or []
    if recent:
        lines.append("Recent activity (most recent last):")
        for e in recent[-6:]:
            lines.append(f"  - {e['seconds_before']:.0f}s ago [{e['app']}] "
                         f"{(e.get('summary') or e.get('window') or '')[:120]}")
    return "\n".join(lines)


def _profile_block(card: ProfileCard | None) -> str:
    if not card or not card.text.strip():
        return "USER PROFILE\n(none available)"
    return f"USER PROFILE\n{card.text}"


def build_messages(inst: TaskInstance, card: ProfileCard | None) -> list[dict]:
    obs = _observation_block(inst)
    prof = _profile_block(card)

    if inst.task_type in ("timing_annotated", "timing_production"):
        if inst.task_type == "timing_annotated":
            question = ("Does this user need assistance at this moment? Consider "
                        "whether a proactive interruption would be welcome or "
                        "annoying.")
        else:
            question = ("Is this user about to start a conversation with their "
                        "assistant within the next 5 minutes?")
        schema = ('{"need_help": true|false, "help_probability": 0.0-1.0, '
                  '"rationale": "one short sentence"}')
        task = f"{question}\n\nReturn JSON:\n{schema}"

    elif inst.task_type == "intention":
        task = ("The user needs assistance. Classify their intention into exactly "
                "one of these categories:\n"
                + ", ".join(INTENTIONS)
                + '\n\nReturn JSON:\n{"intention": "<one category>", '
                  '"rationale": "one short sentence"}')

    elif inst.task_type == "suggestion":
        task = ("Propose the 3 things this user is most likely to want to do or "
                "ask next. Be specific to this user, not generic.\n\n"
                'Return JSON:\n{"suggestions": ["...", "...", "..."], '
                '"rationale": "one short sentence"}')
    else:
        raise ValueError(f"unsupported task type {inst.task_type}")

    user = f"CURRENT OBSERVATION\n{obs}\n\n{prof}\n\nTASK\n{task}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str):
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # first balanced object anywhere in the string
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    start = None
    return None


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "是"):
            return True
        if s in ("false", "no", "0", "否"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def parse(text: str, task_type: str) -> Prediction:
    pred = Prediction(raw=text)
    obj = _extract_json(text)
    if obj is None or not isinstance(obj, dict):
        pred.parse_error = "no JSON object found"
        return pred

    pred.rationale = str(obj.get("rationale") or "")[:500]

    if task_type in ("timing_annotated", "timing_production"):
        nh = _coerce_bool(obj.get("need_help"))
        prob = obj.get("help_probability")
        try:
            prob = float(prob) if prob is not None else None
            if prob is not None:
                prob = min(max(prob, 0.0), 1.0)
        except (TypeError, ValueError):
            prob = None
        if nh is None and prob is not None:
            nh = prob >= 0.5
        if nh is None:
            pred.parse_error = "missing need_help"
            return pred
        # keep a usable score for AUPRC even when the model omits a probability
        pred.need_help, pred.help_probability = nh, (prob if prob is not None
                                                     else float(nh))
        pred.parse_ok = True

    elif task_type == "intention":
        it = str(obj.get("intention") or "").strip()
        if it not in INTENTIONS:
            match = next((c for c in INTENTIONS if c in it), None)
            if match is None:
                pred.parse_error = f"unknown intention {it!r}"
                return pred
            it = match
        pred.intention, pred.parse_ok = it, True

    elif task_type == "suggestion":
        sug = obj.get("suggestions")
        if isinstance(sug, str):
            sug = [sug]
        if not isinstance(sug, list) or not sug:
            pred.parse_error = "missing suggestions"
            return pred
        pred.suggestions = [str(s).strip() for s in sug if str(s).strip()][:5]
        pred.parse_ok = bool(pred.suggestions)

    return pred
