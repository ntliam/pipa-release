"""Memory / profile providers - the baseline ladder B0-B7 plus controls.

Every provider answers the same question in a different representation:
*how do you compress this user's history into something an agent can read?*

    B0 none          total compression
    B1 raw           none - last-k events verbatim
    B2 retrieval     selective - top-k similar past events
    B3 statistical   numeric aggregates
    B4 kg            symbolic priors P(help|app), P(intent|app), transitions
    B5 llm_summary   natural-language summary (needs an LLM; cached)
    B6 deployed      the product's own L4 profile card
    B7 = B3/B5 + B2 (composed via CompositeMemory)

Controls: shuffled, other_user, generic, oracle.

Contract: `build(instance, history)` may read ONLY instances in `history`,
which TaskSet guarantees are strictly earlier and same-user. Controls
deliberately violate the same-user part - that is what they test.
"""
from __future__ import annotations

import collections
import random
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tasks import TaskInstance, TaskSet


@dataclass
class ProfileCard:
    text: str                      # what goes into the prompt
    kind: str                      # provider name
    n_history: int = 0             # how many past events informed it
    meta: dict | None = None

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class MemoryProvider(Protocol):
    name: str

    def build(self, instance: TaskInstance, history: list[TaskInstance]) -> ProfileCard:
        ...

    def select_history(self, instance: TaskInstance,
                       history: list[TaskInstance]) -> list[TaskInstance]:
        """The history this condition is entitled to see.

        Critical for ReAct: tools must be backed by the SAME history the static
        profile was built from. Otherwise a control would render a stranger's
        profile while its tools quietly query the real user - which would
        destroy the control.
        """
        ...


def _default_select(instance, history):
    return history


# --------------------------------------------------------------------------
# B0 / B1
# --------------------------------------------------------------------------


class NoMemory:
    select_history = staticmethod(_default_select)
    name = "b0_none"

    def build(self, instance, history) -> ProfileCard:
        return ProfileCard("", self.name, 0)


class RawHistory:
    """Paste the last k events in verbatim. The 'does structure help at all?'
    baseline - with long-context models this is a real threat to the premise."""
    select_history = staticmethod(_default_select)
    name = "b1_raw"

    def __init__(self, k: int = 20, max_chars: int = 4000):
        self.k, self.max_chars = k, max_chars

    def build(self, instance, history) -> ProfileCard:
        lines = []
        for h in history[-self.k:]:
            # Past outcomes are observable by t0 and every other condition
            # gets them (b2 inline, b3 as a rate, b4 as priors). Omitting them
            # here would make b1 a test of label access, not of representation.
            lines.append(f"- [{h.t0:%m-%d %H:%M}] {_describe(h)}{_outcome(h)}")
        text = "\n".join(lines)[: self.max_chars]
        return ProfileCard(text, self.name, len(history[-self.k:]))


# --------------------------------------------------------------------------
# B2 retrieval
# --------------------------------------------------------------------------


class RetrievalMemory:
    """Top-k past events most similar to the current observation.

    TF-IDF over character n-grams: deterministic, no network, works for mixed
    Chinese/English. Fitted per-instance on that user's own history only, so no
    corpus-level information crosses the chronological boundary.
    """
    select_history = staticmethod(_default_select)
    name = "b2_retrieval"

    def __init__(self, k: int = 5, min_history: int = 3):
        self.k, self.min_history = k, min_history

    def build(self, instance, history) -> ProfileCard:
        if len(history) < self.min_history:
            return ProfileCard("", self.name, len(history))
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [_describe(h) for h in history]
        query = _describe(instance)
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
            mat = vec.fit_transform(docs + [query])
        except ValueError:
            return ProfileCard("", self.name, len(history))
        sims = (mat[:-1] @ mat[-1].T).toarray().ravel()
        top = np.argsort(-sims)[: self.k]
        lines = []
        for rank, idx in enumerate(top, 1):
            h = history[int(idx)]
            lines.append(f"{rank}. (sim={sims[idx]:.2f}) [{h.t0:%m-%d %H:%M}] "
                         f"{_describe(h)}{_outcome(h)}")
        return ProfileCard("\n".join(lines), self.name, len(history),
                           {"max_sim": float(sims.max())})


# --------------------------------------------------------------------------
# B3 statistical
# --------------------------------------------------------------------------


class StatisticalProfile:
    """Deterministic aggregates - no LLM, no training, no labels required
    beyond what was already observed in the past."""
    select_history = staticmethod(_default_select)
    name = "b3_statistical"

    def __init__(self, min_history: int = 3):
        self.min_history = min_history

    def build(self, instance, history) -> ProfileCard:
        if len(history) < self.min_history:
            return ProfileCard("", self.name, len(history))
        apps = collections.Counter()
        hours = collections.Counter()
        intents = collections.Counter()
        topics = collections.Counter()
        past_asks: list[str] = []
        helped = 0
        labelled = 0
        for h in history:
            app = h.observation.get("app_name")
            if app:
                apps[app] += 1
            hours[h.t0.hour] += 1
            lab = h.target.get("label")
            if isinstance(lab, bool):
                labelled += 1
                helped += int(lab)
            it = h.target.get("intention") or (
                h.target.get("label") if h.task_type == "intention" else None)
            if isinstance(it, str) and it:
                intents[it] += 1
            # On the suggestion task there are no apps, intentions or help
            # labels - the user's own past requests are the signal, and the
            # label of a past instance IS what they went on to ask.
            if h.task_type == "suggestion":
                asked = h.target.get("label")
                if isinstance(asked, str) and asked.strip():
                    asked = asked.strip()
                    # consecutive suggestion events often share the same "next
                    # utterance"; repeating it inflates the profile without
                    # adding information
                    if not past_asks or past_asks[-1] != asked:
                        past_asks.append(asked)
                        topics.update(_content_tokens(asked))

        parts = [f"- Observed events: {len(history)}"]
        if past_asks:
            parts.append("- Recent requests from this user (most recent last):")
            for a in past_asks[-6:]:
                parts.append(f"    · {a[:160]}")
        if topics:
            parts.append("- Recurring topics: " + ", ".join(
                f"{t} ({c})" for t, c in topics.most_common(8) if c > 1))
        if apps:
            parts.append("- Frequent applications: " + ", ".join(
                f"{a} ({c})" for a, c in apps.most_common(5)))
        if hours:
            span = ", ".join(f"{h:02d}:00" for h, _ in hours.most_common(3))
            parts.append(f"- Most active hours: {span}")
        if labelled:
            parts.append(f"- Historical help rate: {helped}/{labelled} "
                         f"({helped / labelled:.0%})")
        if intents:
            parts.append("- Recurring intentions: " + ", ".join(
                f"{i} ({c})" for i, c in intents.most_common(5)))
        return ProfileCard("\n".join(parts), self.name, len(history))


# --------------------------------------------------------------------------
# B4 knowledge graph
# --------------------------------------------------------------------------


class KGMemory:
    """Symbolic priors, fitted online on the user's own prior history.

    Fitting online rather than on a global train split keeps B4 under the same
    chronological contract as every other provider. It does use past *labels*,
    which B3/B5 do not - state that when reporting.
    """
    select_history = staticmethod(_default_select)
    name = "b4_kg"

    def __init__(self, min_history: int = 5, laplace: float = 1.0):
        self.min_history, self.laplace = min_history, laplace

    def build(self, instance, history) -> ProfileCard:
        if len(history) < self.min_history:
            return ProfileCard("", self.name, len(history))
        app_help: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        app_intent: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        kw_intent: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        transitions: collections.Counter = collections.Counter()
        prev_app = None
        for h in history:
            app = h.observation.get("app_name") or "?"
            lab = h.target.get("label")
            if isinstance(lab, bool):
                app_help[app][int(lab)] += 1
            intent = h.target.get("intention") or (
                h.target.get("label") if h.task_type == "intention" else None)
            if isinstance(intent, str) and intent:
                app_intent[app][intent] += 1
                for kw in _keywords(h.observation.get("window_title", "")):
                    kw_intent[kw][intent] += 1
            if prev_app and prev_app != app:
                transitions[(prev_app, app)] += 1
            prev_app = app

        cur_app = instance.observation.get("app_name") or "?"
        lines = []
        if cur_app in app_help:
            no, yes = app_help[cur_app]
            p = (yes + self.laplace) / (yes + no + 2 * self.laplace)
            lines.append(f"- P(needs help | app={cur_app}) = {p:.2f} "
                         f"(from {yes + no} past events)")
        if cur_app in app_intent and app_intent[cur_app]:
            tot = sum(app_intent[cur_app].values())
            top = ", ".join(f"{i} {c / tot:.0%}"
                            for i, c in app_intent[cur_app].most_common(3))
            lines.append(f"- P(intent | app={cur_app}): {top}")
        cur_kws = _keywords(instance.observation.get("window_title", ""))
        kw_hits = collections.Counter()
        for kw in cur_kws:
            kw_hits.update(kw_intent.get(kw, {}))
        if kw_hits:
            tot = sum(kw_hits.values())
            lines.append("- P(intent | window keywords): " + ", ".join(
                f"{i} {c / tot:.0%}" for i, c in kw_hits.most_common(3)))
        incoming = [(a, c) for (a, b), c in transitions.items() if b == cur_app]
        if incoming:
            incoming.sort(key=lambda x: -x[1])
            lines.append("- Typically preceded by: " + ", ".join(
                f"{a} ({c})" for a, c in incoming[:3]))
        return ProfileCard("\n".join(lines), self.name, len(history))


# --------------------------------------------------------------------------
# B6 deployed profile
# --------------------------------------------------------------------------

_PROFILE_RE = re.compile(r"<user_profile>(.*?)</user_profile>", re.S)


class DeployedProfile:
    """The product's own L4 card, as of the latest snapshot before t0.

    Only available for production users. This is the shipped-system baseline -
    the one a competitor cannot reproduce.
    """
    select_history = staticmethod(_default_select)
    name = "b6_deployed"

    def __init__(self):
        self._by_user: dict[str, list[tuple]] | None = None

    def _load(self):
        if self._by_user is not None:
            return
        import pandas as pd
        from tasks import DATASET
        t = pd.read_parquet(DATASET / "production" / "texts_gateway.parquet",
                            columns=["user_id", "timestamp", "user_text"])
        t = t[t.user_text.str.contains("<user_profile>", na=False)]
        t["ts"] = pd.to_datetime(t["timestamp"], errors="coerce")
        out: dict[str, list[tuple]] = {}
        for r in t.itertuples():
            m = _PROFILE_RE.search(r.user_text)
            if not m or r.ts is None:
                continue
            out.setdefault(r.user_id, []).append((r.ts.to_pydatetime(), m.group(1).strip()))
        for v in out.values():
            v.sort(key=lambda x: x[0])
        self._by_user = out

    def build(self, instance, history) -> ProfileCard:
        self._load()
        snaps = self._by_user.get(instance.user_id, [])
        latest = None
        for ts, text in snaps:
            if ts < instance.t0:
                latest = text
            else:
                break
        if not latest:
            return ProfileCard("", self.name, 0, {"available": False})
        return ProfileCard(latest[:4000], self.name, len(history), {"available": True})


class RepertoireMemory:
    """The user's distinct past requests, with how often and how recently.

    The suggestion task is selection, not generation: 86% of targets are >=0.55
    similar to something the user already asked, and perfectly selecting 3 of
    their past 40 requests scores 0.869 pass@3 against 0.257 for free
    generation. Structuring the repertoire lets a model rank within it instead
    of inventing requests the user would never make.

    Frequency and recency are stated so the ranker can weigh a recurring need
    against a recent one, rather than defaulting to whichever is newest.
    """
    select_history = staticmethod(_default_select)
    name = "b11_repertoire"

    def __init__(self, k: int = 25, max_chars: int = 2200, min_history: int = 3):
        self.k, self.max_chars, self.min_history = k, max_chars, min_history

    def build(self, instance, history) -> ProfileCard:
        if len(history) < self.min_history:
            return ProfileCard("", self.name, len(history))
        seen: dict[str, dict] = {}
        for pos, h in enumerate(history):
            lab = h.target.get("label")
            if not (isinstance(lab, str) and lab.strip()):
                continue
            key = lab.strip()
            rec = seen.setdefault(key, {"n": 0, "last_pos": pos, "last_t": h.t0})
            rec["n"] += 1
            rec["last_pos"] = pos
            rec["last_t"] = h.t0
        if not seen:
            return ProfileCard("", self.name, len(history))

        total = len(history)
        # rank by recency and repetition together; either alone loses cases
        ranked = sorted(seen.items(),
                        key=lambda kv: (-(kv[1]["n"] > 1), -kv[1]["last_pos"]))
        # The instruction lives in the memory, not the task prompt, so the
        # task block stays byte-identical across conditions. Diagnosis: the
        # answer is among these candidates 74% of the time, our ordering puts
        # it first in 57% of those - but the model only reproduced a listed
        # item in 39% of cases, paraphrasing instead and scoring below
        # threshold. Telling it that reuse is allowed converts coverage into
        # hits.
        lines = ["THIS USER'S PAST REQUESTS (most recent first; "
                 "'x N' = asked N times).",
                 "This user often wants the same things again. If one of these "
                 "fits the current moment, propose it AS WRITTEN rather than "
                 "rephrasing it.", ""]
        for text, rec in ranked[: self.k]:
            ago = total - 1 - rec["last_pos"]
            mult = f" x{rec['n']}" if rec["n"] > 1 else ""
            lines.append(f"- [{ago} events ago{mult}] {text[:160]}")
        return ProfileCard("\n".join(lines)[: self.max_chars], self.name,
                           len(history),
                           {"n_distinct": len(seen),
                            "n_recurring": sum(1 for v in seen.values() if v["n"] > 1)})


# --------------------------------------------------------------------------
# composition and controls
# --------------------------------------------------------------------------


class CompositeMemory:
    """B7: profile + evidence."""
    select_history = staticmethod(_default_select)

    def __init__(self, *providers: MemoryProvider, name: str = "b7_composite"):
        self.providers = providers
        self.name = name

    def build(self, instance, history) -> ProfileCard:
        blocks, n = [], 0
        for p in self.providers:
            card = p.build(instance, history)
            if card:
                blocks.append(f"[{p.name}]\n{card.text}")
                n = max(n, card.n_history)
        return ProfileCard("\n\n".join(blocks), self.name, n)


class ShuffledControl:
    """Same provider, but built from a DIFFERENT user's history.

    Identical format and comparable length - only the identity is wrong. The
    single most important control in the project.
    """

    def __init__(self, inner: MemoryProvider, taskset: TaskSet, seed: int = 42,
                 mode: str = "shuffled"):
        self.inner, self.taskset, self.mode = inner, taskset, mode
        self.name = f"ctrl_{mode}_{inner.name}"
        self._rng = random.Random(seed)
        self._pool: dict[str, list[TaskInstance]] = {}
        for inst in taskset.instances:
            self._pool.setdefault(inst.user_id, []).append(inst)
        for v in self._pool.values():
            v.sort(key=lambda i: i.t0)

    def select_history(self, instance, history):
        others = [u for u in self._pool if u != instance.user_id]
        if not others:
            return []
        # deterministic per instance so runs are reproducible
        rng = random.Random(f"{self.mode}:{instance.task_id}")
        donor = rng.choice(others)
        pool = self._pool[donor]
        want = len(history) or 1
        if self.mode == "shuffled":
            if len(pool) >= want:
                picked = rng.sample(pool, want)
            else:
                # Donor has fewer events than the target. Sampling only what
                # exists makes the control profile systematically shorter,
                # which would confound identity with prompt length - so draw
                # with replacement to match volume.
                picked = [rng.choice(pool) for _ in range(want)]
            picked.sort(key=lambda i: i.t0)
        else:  # other_user: a coherent prefix of a real different person
            picked = pool[:want]
            if len(picked) < want and pool:
                picked = (pool * (want // len(pool) + 1))[:want]
        return picked

    def build(self, instance, history) -> ProfileCard:
        picked = self.select_history(instance, history)
        if not picked:
            return ProfileCard("", self.name, 0)
        card = self.inner.build(instance, picked)
        return ProfileCard(card.text, self.name, len(picked),
                           {"donor": picked[0].user_id, "mode": self.mode})


class GenericControl:
    """One population-level profile for everyone. Separates 'personalisation
    helps' from 'any profile-shaped context helps'."""

    def __init__(self, inner: MemoryProvider, taskset: TaskSet, n_sample: int = 400,
                 seed: int = 42):
        self.inner, self.name = inner, f"ctrl_generic_{inner.name}"
        rng = random.Random(seed)
        pool = rng.sample(taskset.instances, min(n_sample, len(taskset.instances)))
        pool.sort(key=lambda i: i.t0)
        self._pool_hist = pool[:-1]
        self._card = inner.build(pool[-1], pool[:-1])

    def select_history(self, instance, history):
        return self._pool_hist

    def build(self, instance, history) -> ProfileCard:
        return ProfileCard(self._card.text, self.name, self._card.n_history)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _describe(inst: TaskInstance) -> str:
    o = inst.observation
    if inst.task_type in ("timing_annotated", "intention"):
        return (f"{o.get('app_name', '?')} | {o.get('window_title', '')[:60]} | "
                f"{(o.get('screen_summary') or '')[:160]}")
    if inst.task_type == "timing_production":
        return (o.get("screen_summary") or "")[:200]
    return (o.get("context") or "")[:200]


def _outcome(inst: TaskInstance) -> str:
    """Past outcomes are observable by t0 - this is evidence, not leakage."""
    lab = inst.target.get("label")
    if isinstance(lab, bool):
        s = " -> needed help" if lab else " -> no help needed"
        if lab and inst.target.get("intention"):
            s += f" ({inst.target['intention']})"
        return s
    if inst.task_type == "intention":
        return f" -> intent: {lab}"
    if inst.task_type == "suggestion" and isinstance(lab, str) and lab.strip():
        # What the user actually asked next IS the observed outcome here.
        # Returning "" made every memory except b3_statistical blind to it,
        # since b3 reads target["label"] through its own path - an information
        # asymmetry, not a representational difference. b3 scored 0.439 pass@3
        # against ~0.22 for everything else purely because of this.
        return f" -> user then asked: {lab.strip()[:120]}"
    return ""


_KW_SPLIT = re.compile(r"[\s\-_|·,，。.:：/\\()\[\]]+")

# CJK runs of 2-4 chars plus latin words. Crude, but deterministic and enough
# to surface "this user keeps asking about X" without a segmenter dependency.
_CJK = re.compile(r"[一-鿿]{2,4}")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{2,}")
_STOP = {"用户", "输入", "帮我", "一个", "可以", "什么", "怎么", "如何", "这个",
         "我们", "现在", "the", "and", "for", "with", "you", "your", "please"}


def _content_tokens(text: str, limit: int = 12) -> list[str]:
    if not text:
        return []
    toks = _CJK.findall(text) + [w.lower() for w in _LATIN.findall(text)]
    return [t for t in toks if t not in _STOP][:limit]


def _keywords(title: str, min_len: int = 2, max_n: int = 6) -> list[str]:
    if not title:
        return []
    toks = [t.lower() for t in _KW_SPLIT.split(title) if len(t) >= min_len]
    return toks[:max_n]


def default_providers(taskset: TaskSet, include_controls: bool = True,
                      engine=None) -> dict:
    """Build the condition set. `engine` enables the LLM-based reconstructor
    (b5); without it that condition is omitted rather than silently empty."""
    stat = StatisticalProfile()
    retr = RetrievalMemory()
    providers = {
        "b0_none": NoMemory(),
        "b1_raw": RawHistory(),
        "b2_retrieval": retr,
        "b3_statistical": stat,
        "b4_kg": KGMemory(),
        "b7_composite": CompositeMemory(stat, retr),
    }
    if engine is not None:
        from baselines import GenerativeAgentsReflection, RecursiveSummaryMemory
        # prior-work baselines, same information access and rebuild schedule
        providers["b9_genagents"] = GenerativeAgentsReflection(engine)
        providers["b10_summary"] = RecursiveSummaryMemory(engine)
        from profile import ProfileReconstructor
        recon = ProfileReconstructor(engine)
        # b5 alone answers the central question: how much of the identity
        # ceiling does a reconstructed profile recover on its own?
        providers["b5_reconstructed"] = recon
        # b8 is the full PIPA system: reconstructed profile + precedent.
        providers["b8_pipa"] = CompositeMemory(recon, retr, name="b8_pipa")
        # PIPA as a personalised ranker: the profile says which needs recur and
        # when, the repertoire supplies the candidates to rank.
        rep = RepertoireMemory()
        providers["b11_repertoire"] = rep
        providers["b12_pipa_rank"] = CompositeMemory(recon, rep,
                                                     name="b12_pipa_rank")
        # v3: LLM proposes cues, code measures support. Its identity control
        # gets its OWN instance so donor profiles can never share a cache
        # entry with real ones.
        from rules import RuleProfile
        # Binary-label tasks only - see RuleProfile.SUPPORTED. Registering it
        # elsewhere produced an empty card that scored identically to b0_none.
        if taskset.task_type in RuleProfile.SUPPORTED:
            providers["b13_rules"] = RuleProfile(engine, taskset)
            if include_controls:
                providers["ctrl_rules_other_user"] = ShuffledControl(
                    RuleProfile(engine, taskset), taskset, mode="other_user")
    if taskset.task_type in ("timing_production", "suggestion"):
        providers["b6_deployed"] = DeployedProfile()
    if include_controls:
        providers["ctrl_shuffled"] = ShuffledControl(stat, taskset, mode="shuffled")
        providers["ctrl_other_user"] = ShuffledControl(stat, taskset, mode="other_user")
        # STRUCTURALLY MATCHED controls. The three above wrap the statistical
        # profile only, so comparing them against b7_composite (profile +
        # retrieval) confounds "wrong user" with "no retrieval". These wrap the
        # same composite provider, so the ONLY difference from b7 is whose
        # history it is - which is the personalisation claim itself.
        composite = CompositeMemory(stat, retr)
        providers["ctrl_shuffled_matched"] = ShuffledControl(
            composite, taskset, mode="shuffled")
        providers["ctrl_other_user_matched"] = ShuffledControl(
            composite, taskset, mode="other_user")
        providers["ctrl_generic"] = GenericControl(stat, taskset)
    return providers
