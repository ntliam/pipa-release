"""ReAct agent: the model decides what to look up about the user.

The key design choice is that **the action space mirrors the memory
representation**. In single-shot mode a condition receives a fixed profile
block; in ReAct mode it receives *tools that query the same memory*. So the
comparison isolates one thing: does letting the agent choose what to retrieve
beat handing it a fixed summary?

    condition        single-shot                ReAct tools
    b0_none          (nothing)                  (none - pure reflection)
    b1_raw           last-20 events dumped      get_recent_events
    b2_retrieval     fixed top-5 similar        retrieve_similar(query)
    b3_statistical   fixed aggregate card       get_user_stats(dimension)
    b4_kg            fixed priors               get_priors(app)
    b7_composite     both, fixed                retrieve_similar + get_user_stats

Tools are backed by `provider.select_history()`, NOT the raw history. A
shuffled control therefore queries the donor's events, so its tools cannot
smuggle in the real user's data.

Every Thought/Action/Observation is logged for the qualitative analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import agent as agent_mod
import memory as mem
from tasks import TaskInstance

MAX_OBS_CHARS = 1200


@dataclass
class Tool:
    name: str
    description: str
    args: str
    fn: Callable[..., str]

    def spec(self) -> str:
        return f"- {self.name}({self.args}): {self.description}"


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    error: str | None = None


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finished: bool = False
    stop_reason: str = ""

    def to_json(self) -> list[dict]:
        return [{"thought": s.thought, "action": s.action,
                 "action_input": s.action_input,
                 "observation": s.observation[:400], "error": s.error}
                for s in self.steps]

    @property
    def actions_used(self) -> list[str]:
        return [s.action for s in self.steps if s.action]


# --------------------------------------------------------------------------
# tool construction
# --------------------------------------------------------------------------


def _fmt_event(h: TaskInstance) -> str:
    return f"[{h.t0:%m-%d %H:%M}] {mem._describe(h)}{mem._outcome(h)}"


def build_tools(instance: TaskInstance, history: list[TaskInstance],
                provider_name: str) -> list[Tool]:
    """Tools mirroring the provider's memory representation."""
    tools: list[Tool] = []

    def retrieve_similar(query: str = "", k: int = 5) -> str:
        if not history:
            return "No past activity recorded for this user."
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [mem._describe(h) for h in history]
        q = query.strip() or mem._describe(instance)
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
            m = vec.fit_transform(docs + [q])
        except ValueError:
            return "Retrieval unavailable (insufficient text)."
        sims = (m[:-1] @ m[-1].T).toarray().ravel()
        k = max(1, min(int(k or 5), 10))
        top = np.argsort(-sims)[:k]
        return "\n".join(f"{i}. (sim={sims[j]:.2f}) {_fmt_event(history[int(j)])}"
                         for i, j in enumerate(top, 1))

    def get_user_stats(dimension: str = "all") -> str:
        card = mem.StatisticalProfile(min_history=1).build(instance, history)
        if not card.text:
            return "No statistics available (no past activity)."
        if dimension and dimension != "all":
            hits = [ln for ln in card.text.splitlines()
                    if dimension.lower() in ln.lower()]
            if hits:
                return "\n".join(hits)
        return card.text

    def get_priors(app: str = "") -> str:
        probe = instance
        if app:
            probe = TaskInstance(
                task_id=instance.task_id, task_type=instance.task_type,
                user_id=instance.user_id, t0=instance.t0,
                observation={**instance.observation, "app_name": app},
                target=instance.target)
        card = mem.KGMemory(min_history=1).build(probe, history)
        return card.text or f"No priors recorded for app={app or 'current'}."

    def get_recent_events(k: int = 10) -> str:
        if not history:
            return "No past activity recorded for this user."
        k = max(1, min(int(k or 10), 25))
        return "\n".join(_fmt_event(h) for h in history[-k:])

    catalog = {
        "retrieve_similar": Tool(
            "retrieve_similar",
            "Find this user's most similar past situations, with what happened.",
            'query: str, k: int = 5', retrieve_similar),
        "get_user_stats": Tool(
            "get_user_stats",
            "Aggregate statistics about this user: apps, active hours, "
            "historical help rate, recurring intentions.",
            'dimension: str = "all"', get_user_stats),
        "get_priors": Tool(
            "get_priors",
            "Learned probabilities for this user, e.g. P(needs help | app).",
            'app: str = ""', get_priors),
        "get_recent_events": Tool(
            "get_recent_events",
            "This user's most recent activity in chronological order.",
            'k: int = 10', get_recent_events),
    }

    wanted = {
        "b0_none": [],
        "b6_deployed": [],
        "b1_raw": ["get_recent_events"],
        "b2_retrieval": ["retrieve_similar"],
        "b3_statistical": ["get_user_stats"],
        "b4_kg": ["get_priors"],
        "b7_composite": ["retrieve_similar", "get_user_stats"],
    }
    # controls wrap an inner provider - give them the inner provider's tools so
    # the only difference from the real condition remains whose data it is
    base = provider_name
    if base.startswith("ctrl_"):
        base = "b3_statistical"
    for key in wanted.get(base, ["retrieve_similar", "get_user_stats"]):
        tools.append(catalog[key])
    return tools


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

REACT_SYSTEM = (
    "You are a proactive personal assistant. You reason step by step and call "
    "tools to look up what you know about this specific user before deciding.\n\n"
    "Always begin with a brief thought. When tools are available, call at least "
    "one to ground your decision in this user's history before giving a final "
    "answer - you are not given their profile up front.\n\n"
    "Respond with a SINGLE JSON object and nothing else. Either:\n"
    '  {"thought": "...", "action": "<tool>", "action_input": {...}}\n'
    "to call a tool, or:\n"
    '  {"thought": "...", "final_answer": {...}}\n'
    "to finish. Do not wrap the JSON in markdown fences."
)

# When the condition exposes tools, the profile must NOT be pre-rendered -
# otherwise the agent already has the answer and never acts, which silently
# turns ReAct back into single-shot (observed: 16/16 instances, 0 tool calls).
PROFILE_PLACEHOLDER = (
    "USER PROFILE\n(not loaded - use the tools below to look up what you know "
    "about this user)")


def _render_tools(tools: list[Tool]) -> str:
    if not tools:
        return ("AVAILABLE TOOLS\n(none - reason from the observation alone, "
                "then give final_answer)")
    return "AVAILABLE TOOLS\n" + "\n".join(t.spec() for t in tools)


class ReactAgent:
    def __init__(self, engine, max_steps: int = 3):
        self.engine = engine
        self.max_steps = max_steps

    def run(self, instance: TaskInstance, history: list[TaskInstance],
            provider) -> tuple[agent_mod.Prediction, Trace]:
        # tools see exactly what the provider is entitled to see
        allowed = provider.select_history(instance, history)
        tools = build_tools(instance, allowed, provider.name)
        by_name = {t.name: t for t in tools}

        card = provider.build(instance, history)
        base = agent_mod.build_messages(instance, card)
        # identical observation + task block as single-shot; only the scaffold
        # and the tool catalogue differ
        user_block = base[1]["content"]
        if tools:
            # swap the rendered profile for a placeholder so the agent has to
            # fetch it. Conditions with no tools (b0_none, b6_deployed) keep
            # their static block - there is nothing for them to fetch.
            head, _, tail = user_block.partition("USER PROFILE")
            _, _, after = tail.partition("\n\nTASK")
            user_block = f"{head}{PROFILE_PLACEHOLDER}\n\nTASK{after}"
        messages = [
            {"role": "system", "content": REACT_SYSTEM},
            {"role": "user", "content": f"{user_block}\n\n{_render_tools(tools)}"},
        ]

        trace = Trace()
        final_text = ""
        for step_i in range(self.max_steps + 1):
            last = step_i == self.max_steps
            if last and tools:
                messages.append({
                    "role": "user",
                    "content": "Step limit reached. Give your final_answer now.",
                })
            resp = self.engine.complete(messages, tag=f"react:{instance.task_id}:{step_i}")
            trace.n_llm_calls += 1
            trace.prompt_tokens += resp.prompt_tokens
            trace.completion_tokens += resp.completion_tokens
            trace.latency_ms += resp.latency_ms
            if resp.error == "dry_run":
                trace.stop_reason = "dry_run"
                return agent_mod.Prediction(parse_error="dry_run"), trace
            if not resp.ok:
                trace.stop_reason = f"llm_error: {resp.error}"
                return agent_mod.Prediction(parse_error=resp.error), trace

            obj = agent_mod._extract_json(resp.text)
            if not isinstance(obj, dict):
                step = Step(observation="", error="unparseable step")
                trace.steps.append(step)
                messages.append({"role": "assistant", "content": resp.text[:600]})
                messages.append({"role": "user", "content":
                                 "That was not valid JSON. Reply with a single "
                                 "JSON object as specified."})
                continue

            thought = str(obj.get("thought") or "")[:600]

            if "final_answer" in obj or "action" not in obj:
                payload = obj.get("final_answer", obj)
                final_text = json.dumps(payload, ensure_ascii=False)
                trace.steps.append(Step(thought=thought, action="final_answer"))
                trace.finished = True
                trace.stop_reason = "final_answer"
                break

            name = str(obj.get("action") or "").strip()
            args = obj.get("action_input") or {}
            if not isinstance(args, dict):
                args = {"query": str(args)}
            step = Step(thought=thought, action=name, action_input=args)

            tool = by_name.get(name)
            if tool is None:
                step.error = f"unknown tool {name!r}"
                step.observation = (f"Error: no such tool. Available: "
                                    f"{list(by_name) or 'none'}. "
                                    f"Give final_answer instead.")
            else:
                try:
                    step.observation = str(tool.fn(**args))[:MAX_OBS_CHARS]
                except TypeError as exc:
                    step.error = f"bad arguments: {exc}"
                    step.observation = f"Error calling {name}: {exc}"
                except Exception as exc:  # noqa: BLE001
                    step.error = f"{type(exc).__name__}: {exc}"
                    step.observation = f"Error calling {name}: {exc}"
            trace.steps.append(step)
            messages.append({"role": "assistant", "content": resp.text[:600]})
            messages.append({"role": "user",
                             "content": f"OBSERVATION\n{step.observation}"})
        else:
            trace.stop_reason = "step_limit"

        pred = agent_mod.parse(final_text, instance.task_type)
        return pred, trace
