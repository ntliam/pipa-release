"""Memory baselines from prior work, implemented faithfully.

Two named methods reviewers will expect, both given the SAME information
access as PIPA (past outcomes included), the SAME rebuild schedule (bucketed
prefixes) and comparable length budgets. Handicapping a baseline is the
easiest way to lose a paper, so the defaults here mirror ours deliberately.

  GenerativeAgentsReflection  - Park et al., "Generative Agents". Memory stream
      scored by recency + importance + relevance, plus periodic reflection that
      synthesises observations into higher-level statements about the person.
      This is the closest prior work to profile reconstruction and therefore
      the most important comparison in the project.

  RecursiveSummaryMemory      - MemGPT-style. Maintains a running summary,
      folding new events into the previous summary. Distinguishes
      reconstruction-as-inference from plain summarisation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from memory import ProfileCard, _describe, _outcome
from profile import BUCKETS, _bucket, _render_events
from tasks import TaskInstance

# --------------------------------------------------------------------------
# Generative Agents
# --------------------------------------------------------------------------

REFLECT_SYSTEM = (
    "You observe a person's activity and periodically step back to form "
    "higher-level insights about them. Insights should be general statements "
    "that go beyond any single observation."
)

REFLECT_TEMPLATE = """Recent observations about this person:

{events}

What are the 3 most salient high-level insights you can infer about this
person from these observations? Write each as one sentence. Only output the
three statements, numbered."""

IMPORTANCE_SYSTEM = (
    "You rate how informative each observation is for understanding a person. "
    "1 = purely routine, 10 = highly revealing of who they are."
)


@dataclass
class _Reflection:
    text: str
    n_events: int


class GenerativeAgentsReflection:
    """Park et al. memory stream: retrieval by recency + importance +
    relevance, with periodic reflection.

    Importance is LLM-rated in batches and cached, since rating every
    observation individually would cost one call per event.
    """

    name = "b9_genagents"

    def __init__(self, engine=None, k: int = 5, alpha_recency: float = 1.0,
                 alpha_importance: float = 1.0, alpha_relevance: float = 1.0,
                 decay: float = 0.995, batch: int = 25, max_chars: int = 1600):
        self.engine = engine
        self.k = k
        self.a_rec, self.a_imp, self.a_rel = (alpha_recency, alpha_importance,
                                              alpha_relevance)
        self.decay = decay
        self.batch = batch
        self.max_chars = max_chars
        self._reflect: dict[tuple[str, int], _Reflection] = {}
        self._importance: dict[tuple[str, int], list[float]] = {}

    @staticmethod
    def select_history(instance, history):
        return history

    # -- importance --------------------------------------------------------
    def _rate(self, user_id: str, prefix: list[TaskInstance]) -> list[float]:
        key = (user_id, len(prefix))
        if key in self._importance:
            return self._importance[key]
        scores: list[float] = []
        if self.engine is None:
            scores = [5.0] * len(prefix)
        else:
            for i in range(0, len(prefix), self.batch):
                chunk = prefix[i:i + self.batch]
                listing = "\n".join(
                    f"{j+1}. {_describe(h)[:150]}" for j, h in enumerate(chunk))
                msg = [{"role": "system", "content": IMPORTANCE_SYSTEM},
                       {"role": "user", "content":
                        f"Rate each observation 1-10. Output only "
                        f"'<number>: <rating>' lines.\n\n{listing}"}]
                resp = self.engine.complete(msg, tag=f"importance:{user_id}:{i}")
                vals = [5.0] * len(chunk)
                if resp.ok:
                    for m in re.finditer(r"(\d+)\s*[:.\-]\s*(\d+(?:\.\d+)?)", resp.text):
                        idx, val = int(m.group(1)) - 1, float(m.group(2))
                        if 0 <= idx < len(chunk):
                            vals[idx] = min(max(val, 1.0), 10.0)
                scores.extend(vals)
        self._importance[key] = scores
        return scores

    # -- reflection --------------------------------------------------------
    def _reflection(self, user_id: str, prefix: list[TaskInstance]) -> _Reflection:
        key = (user_id, len(prefix))
        hit = self._reflect.get(key)
        if hit is not None:
            return hit
        text = ""
        if self.engine is not None and prefix:
            msg = [{"role": "system", "content": REFLECT_SYSTEM},
                   {"role": "user", "content": REFLECT_TEMPLATE.format(
                       events=_render_events(prefix, limit=60))}]
            resp = self.engine.complete(msg, tag=f"reflect:{user_id}:{len(prefix)}")
            if resp.ok:
                text = resp.text.strip()
        built = _Reflection(text, len(prefix))
        self._reflect[key] = built
        return built

    # -- retrieval ---------------------------------------------------------
    def build(self, instance: TaskInstance,
              history: list[TaskInstance]) -> ProfileCard:
        if len(history) < 3:
            return ProfileCard("", self.name, len(history))
        n = _bucket(len(history))
        prefix = history[:n] if n >= 8 else history

        importance = self._rate(instance.user_id, prefix)
        reflection = self._reflection(instance.user_id, prefix)

        # relevance
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [_describe(h) for h in history]
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
            mat = vec.fit_transform(docs + [_describe(instance)])
            rel = (mat[:-1] @ mat[-1].T).toarray().ravel()
        except ValueError:
            rel = np.zeros(len(history))

        # recency: exponential decay in event distance
        rec = np.array([self.decay ** (len(history) - 1 - i)
                        for i in range(len(history))])
        imp = np.array([(importance[i] if i < len(importance) else 5.0) / 10.0
                        for i in range(len(history))])

        def norm(v):
            rng = v.max() - v.min()
            return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

        score = (self.a_rec * norm(rec) + self.a_imp * norm(imp)
                 + self.a_rel * norm(rel))
        top = np.argsort(-score)[: self.k]

        lines = []
        if reflection.text:
            lines.append("REFLECTIONS")
            lines.append(reflection.text)
            lines.append("")
        lines.append("RETRIEVED MEMORIES")
        for rank, idx in enumerate(top, 1):
            h = history[int(idx)]
            lines.append(f"{rank}. [{h.t0:%m-%d %H:%M}] {_describe(h)}{_outcome(h)}")
        text = "\n".join(lines)[: self.max_chars]
        return ProfileCard(text, self.name, len(history),
                           {"reflection_events": reflection.n_events})


# --------------------------------------------------------------------------
# Recursive summary
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You maintain a running summary of a person's activity for an assistant. "
    "Fold new information into the existing summary, keeping it compact and "
    "preserving what is useful for anticipating their behaviour."
)

SUMMARY_TEMPLATE = """EXISTING SUMMARY
{prev}

NEW ACTIVITY
{events}

Write the updated summary in under 200 words."""


class RecursiveSummaryMemory:
    """MemGPT-style running summary, folded forward across buckets."""

    name = "b10_summary"

    def __init__(self, engine=None, min_history: int = 8, max_chars: int = 1600):
        self.engine = engine
        self.min_history = min_history
        self.max_chars = max_chars
        self._cache: dict[tuple[str, int], str] = {}

    @staticmethod
    def select_history(instance, history):
        return history

    def _summary(self, user_id: str, prefix: list[TaskInstance]) -> str:
        key = (user_id, len(prefix))
        if key in self._cache:
            return self._cache[key]
        if self.engine is None or not prefix:
            self._cache[key] = ""
            return ""

        # fold forward from the previous bucket so the summary is genuinely
        # recursive rather than a one-shot summarisation of everything
        smaller = [b for b in BUCKETS if b < len(prefix)]
        if smaller:
            prev = self._summary(user_id, prefix[:smaller[-1]])
            new_events = prefix[smaller[-1]:]
        else:
            prev, new_events = "(none yet)", prefix

        msg = [{"role": "system", "content": SUMMARY_SYSTEM},
               {"role": "user", "content": SUMMARY_TEMPLATE.format(
                   prev=prev or "(none yet)",
                   events=_render_events(new_events, limit=60))}]
        resp = self.engine.complete(msg, tag=f"summary:{user_id}:{len(prefix)}")
        text = resp.text.strip() if resp.ok else ""
        self._cache[key] = text
        return text

    def build(self, instance: TaskInstance,
              history: list[TaskInstance]) -> ProfileCard:
        n = _bucket(len(history))
        if n < self.min_history:
            return ProfileCard("", self.name, len(history), {"bucket": n})
        text = self._summary(instance.user_id, history[:n])
        if not text:
            return ProfileCard("", self.name, len(history), {"bucket": n})
        return ProfileCard(text[: self.max_chars], self.name, len(history),
                           {"bucket": n, "built_from": n})
