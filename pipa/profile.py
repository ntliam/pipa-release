"""Profile reconstruction — the PIPA contribution.

Compresses a user's behavioural history into a bounded, inspectable
natural-language profile, so the raw history can be discarded.

The distinction that matters: this **infers latent attributes** (working
rhythm, task style, what leads this person to ask for help rather than
self-solve) instead of tallying surface statistics. A surface summary recovers
AUROC 0.674 of an available 0.854 identity ceiling; the 0.18 gap is what
inference has to close.

Two mechanisms keep it honest and affordable:

* **Chronological prefixes.** The profile at instance i is built from a prefix
  of that user's strictly-earlier history, never from future events.
* **Bucketed refresh.** Rebuilding per instance would cost one LLM call per
  instance per user. Profiles are rebuilt at history sizes 8, 16, 32, 64, 128,
  256 - about six builds per user instead of four hundred - which is also how a
  deployed system refreshes a profile. Bucketing is applied to the *prefix
  length*, so it never reaches forward in time.
"""
from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass

from memory import ProfileCard, _describe, _outcome
from tasks import TaskInstance

# Rebuild points. A user with 200 prior events uses the profile built from
# their first 128 - stale by design, and the coverage line says so.
BUCKETS = (8, 16, 32, 64, 128, 256, 512)

MAX_PROFILE_CHARS = 1600

SYSTEM = (
    "You build compact user models for a proactive assistant. You are given a "
    "person's recorded activity, with what happened at each point. Infer what "
    "kind of person they are - not what they did.\n\n"
    "Write CLAIMS ABOUT THE PERSON that would help predict their future "
    "behaviour. Do not reproduce inventories of applications or timestamps: "
    "the assistant already has those. State tendencies, working style, and the "
    "conditions under which they seek help rather than solving things "
    "themselves.\n\n"
    "You MUST quantify how often this person needs assistance, overall and by "
    "situation. Having the model estimate this is deliberate, and was tested: "
    "supplying the exact computed rate in this prompt instead measured WORSE "
    "(AUPRC 0.203 -> 0.165, AUROC 0.695 -> 0.658), apparently because the model "
    "then anchors on the figure and stops reading the current context. The "
    "computed rate is applied in the fusion layer instead, where it is not "
    "degraded by passing through the model.\n\n"
    "Be specific and falsifiable. Prefer 'switches to a browser to look things "
    "up mid-debugging rather than reasoning it out' over 'uses browsers'. If "
    "the evidence is thin, say so rather than inventing detail."
)

TEMPLATE = """ACTIVITY RECORD ({n} events, {span})

{events}

Write a user model with exactly these sections, under 250 words total:

HELP PROPENSITY: overall, roughly what fraction of the time does this person
  need assistance? Then break it down: in which situations is it much higher,
  and in which much lower? State this as calibrated tendencies.
WORKING STYLE: how this person approaches tasks
HELP-SEEKING: what makes them ask for assistance versus continue alone
RHYTHM: when and how they engage
DISTINCTIVE: what separates them from a typical user
CONFIDENCE: what this model is based on, and where it is uncertain
"""


def _bucket(n: int) -> int:
    """Largest rebuild point at or below n; 0 if below the first."""
    usable = [b for b in BUCKETS if b <= n]
    return usable[-1] if usable else 0


def _key(i: int, seed: str) -> float:
    """Deterministic uniform priority in [0,1) for position i."""
    h = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0 ** 64


def sample_indices(n: int, limit: int, seed: str = "") -> list[int]:
    """Uniform sample of range(n) that is MONOTONE in n.

    `np.linspace` redraws from scratch at every rebuild, so consecutive
    profiles are built from largely disjoint evidence: measured carry-over
    across the buckets that matter is 47% / 38% / 35% (64->128->256->512),
    with an input Jaccard of 0.30 / 0.24 / 0.21. Measured profile churn was
    0.36-0.41 trigram overlap, i.e. the text was no more stable than the
    evidence underneath it - the profile churned because its input churned,
    not because the user changed.

    Taking the `limit` lowest priority keys is still a uniform sample, but a
    position can only ever LEAVE the sample as n grows, never re-enter. That
    lifts carry-over to limit*n/m - 50% per doubling, which is the ceiling for
    any sampler that stays uniform over a growing prefix.
    """
    if n <= limit:
        return list(range(n))
    keys = sorted(range(n), key=lambda i: _key(i, seed))
    return sorted(keys[:limit])


def _render_events(history: list[TaskInstance], limit: int = 60,
                   seed: str = "") -> str:
    """A representative sample of the prefix, oldest first.

    Taking the most recent `limit` events instead would make the sample
    unrepresentative as history grows - at a 512-event prefix the model would
    see the last 12% while the prompt claims it is the whole record, and the
    inferred help propensity is then estimated from a narrow recent slice.
    That produced propensities of 85-100% for a user whose true rate was 16%,
    and collapsed the profile from AUROC 0.740 to 0.540.

    Uniform coverage keeps the prompt budget fixed while preserving the base
    rate and the spread of the full prefix; monotonicity keeps consecutive
    rebuilds comparable (see `sample_indices`).
    """
    if len(history) <= limit:
        picked = history
        note = ""
    else:
        idx = sample_indices(len(history), limit, seed)
        picked = [history[i] for i in idx]
        note = (f"(evenly sampled across all {len(history)} events, "
                f"so proportions here reflect the whole record)\n")
    lines = [f"- [{h.t0:%m-%d %H:%M}] {_describe(h)}{_outcome(h)}" for h in picked]
    return note + "\n".join(lines)


def computed_statistics(history: list[TaskInstance]) -> str:
    """Deterministic statistics the backbone cannot estimate reliably.

    Asking the model to infer how often a user needs help gives a correlation
    of only +0.207 with the truth (mean absolute error 18.5 points, bias +6.8):
    LLMs do not count frequencies from examples. Yet that base rate is close to
    the whole identity signal - used directly it reaches AUPRC 0.281 against
    0.203 for the model's own estimate.

    So the countable part is computed and supplied; the model is left to infer
    only what is genuinely qualitative.
    """
    labs = [h.target.get("label") for h in history]
    binary = [bool(x) for x in labs if isinstance(x, bool)]
    lines = []
    if binary:
        pi = sum(binary) / len(binary)
        lines.append(f"- Assistance rate: {pi:.0%} ({sum(binary)} of {len(binary)} "
                     f"past events)")
        # recent window: drift the overall rate cannot show
        tail = binary[-min(50, len(binary)):]
        if len(tail) >= 10:
            lines.append(f"- Recent {len(tail)} events: {sum(tail)/len(tail):.0%}")
        # conditional on time of day, the only context axis present in every task
        buckets: dict[str, list[bool]] = {}
        for h, b in zip([h for h, x in zip(history, labs) if isinstance(x, bool)],
                        binary):
            key = ("morning" if 5 <= h.t0.hour < 12 else
                   "afternoon" if 12 <= h.t0.hour < 18 else
                   "evening" if 18 <= h.t0.hour < 23 else "night")
            buckets.setdefault(key, []).append(b)
        parts = [f"{k} {sum(v)/len(v):.0%}" for k, v in buckets.items()
                 if len(v) >= 8]
        if len(parts) >= 2:
            lines.append("- By time of day: " + ", ".join(parts))

    intents = [x for x in labs if isinstance(x, str) and x]
    if intents and len(set(intents)) > 1:
        from collections import Counter
        top = Counter(intents).most_common(4)
        lines.append("- Most frequent outcomes: " + ", ".join(
            f"{k[:40]} {c/len(intents):.0%}" for k, c in top))
    return "\n".join(lines)


@dataclass
class _Built:
    text: str
    n_events: int


class ProfileReconstructor:
    """LLM-based profile reconstruction. Requires an engine; without one it
    returns an empty card so dry runs still work."""

    name = "b5_reconstructed"

    def __init__(self, engine=None, min_history: int = 8,
                 max_chars: int = MAX_PROFILE_CHARS):
        self.engine = engine
        self.min_history = min_history
        self.max_chars = max_chars
        self._cache: dict[tuple[str, int], _Built] = {}

    @staticmethod
    def select_history(instance, history):
        return history

    # -- construction ------------------------------------------------------
    def _build_prefix(self, user_id: str, prefix: list[TaskInstance]) -> _Built:
        key = (user_id, len(prefix))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        span = "unknown span"
        if prefix:
            days = (prefix[-1].t0 - prefix[0].t0).days
            span = f"{days} day span" if days else "under a day"

        prompt = TEMPLATE.format(n=len(prefix), span=span,
                                 events=_render_events(prefix, seed=user_id))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]

        text = ""
        if self.engine is not None:
            resp = self.engine.complete(messages, tag=f"profile:{user_id}:{len(prefix)}")
            if resp.ok:
                text = _clean_profile(resp.text)[: self.max_chars]

        built = _Built(text, len(prefix))
        self._cache[key] = built
        return built

    def build(self, instance: TaskInstance,
              history: list[TaskInstance]) -> ProfileCard:
        n = _bucket(len(history))
        if n < self.min_history:
            return ProfileCard("", self.name, len(history),
                               {"reason": "insufficient history", "bucket": n})
        built = self._build_prefix(instance.user_id, history[:n])
        if not built.text:
            return ProfileCard("", self.name, len(history), {"bucket": n})

        # Coverage is part of the profile: it lets the agent discount a thin or
        # stale model rather than trusting it uniformly.
        coverage = (f"[model built from this user's first {built.n_events} "
                    f"recorded events; {len(history) - built.n_events} more have "
                    f"occurred since]")
        return ProfileCard(f"{built.text}\n\n{coverage}", self.name,
                           len(history),
                           {"bucket": n, "built_from": built.n_events,
                            "staleness": len(history) - built.n_events})

    # -- diagnostics -------------------------------------------------------
    @property
    def n_profiles_built(self) -> int:
        return len(self._cache)

    def built_profiles(self) -> dict[tuple[str, int], str]:
        return {k: v.text for k, v in self._cache.items()}


_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.M)
_LEAD = re.compile(r"^\s*(here is|here's|certainly|sure)[^\n]*\n", re.I)


def _clean_profile(text: str) -> str:
    text = _FENCE.sub("", text or "").strip()
    text = _LEAD.sub("", text).strip()
    return text
