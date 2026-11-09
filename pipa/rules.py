"""Rule-structured profile reconstruction (PIPA v3).

Same input as `profile.ProfileReconstructor`, different contract about who is
allowed to say what.

The v2 profile asks the backbone for everything, including quantities it cannot
estimate: its stated help-rate correlates only +0.207 with the truth (MAE 18.5
points), and 29% of profiles that cite their own count contradict it by more
than 20 points. A live example from `u_0fd3b728` at bucket 512 opens with
"~65% of observed activity episodes involve help-seeking (33/512 coded as
needed help)" - 33/512 is 6.4%.

Here the split is explicit:

* the model proposes CUES - which recurring situations are worth conditioning
  on for this person. That is the qualitative judgement it is good at, and
  ablation says it is the part that carries the effect (removing the explicit
  when-is-help-needed breakdown drops AUROC 0.769 -> 0.521).
* the code computes SUPPORT - how often help was actually wanted under each
  cue, over the whole prefix rather than the sampled excerpt.

That makes the profile stable for a reason the sampler fix cannot achieve on
its own. Counts are computed over all events, so they do not move when the
excerpt shown to the model changes; only cue *proposal* is sample-dependent.

The scoring contract is residual: the current observation sets the situation,
and the profile only supplies this user's deviation from the population. Users
close to the population norm get a baseline near zero and are left alone -
which is what `corr(atypicality, gain) = +0.616` implies they should be.

This is deliberately close to `b4_kg`, which computes P(help | app) and is the
strongest method on annotated timing (0.885 AUPRC) and at cold start (0.866
AUROC). The difference is that cues are inferred semantic conditions rather
than exact app names, so they generalise across surface forms.
"""
from __future__ import annotations


import json
import math
import re
from dataclasses import dataclass, asdict, field

from memory import ProfileCard, _default_select, _describe
from profile import BUCKETS, _bucket, _render_events
from tasks import TaskInstance, TaskSet

MAX_RULES = 8            # bounded profile: bounded adaptation cost
MIN_SUPPORT = 3          # a cue is not a rule until it has been seen 3 times
ALPHA_RULE = 5.0         # shrinkage toward the population rate, per rule
ALPHA_USER = 30.0        # shrinkage for the user-level baseline
MAX_CARD_CHARS = 1500

SYSTEM = (
    "You read a person's recorded activity and name the recurring SITUATIONS "
    "that predict whether they want a proactive assistant to step in.\n\n"
    "You are NOT asked how often anything happens. Every count is measured "
    "from the full record afterwards; numbers you invent would be discarded "
    "and would only mislead. Name the conditions, not the frequencies.\n\n"
    "A good cue is a situation that recurs and that plausibly separates "
    "wanting help from wanting to be left alone - 'an error or failing test is "
    "visible while editing code' rather than 'uses an IDE'. Cues must be "
    "detectable from the text of a single observation."
)

TEMPLATE = """ACTIVITY RECORD ({n} events, {span})

{events}

Propose up to {k} cues. Return ONLY a JSON array, no prose:

[
  {{"id": "short_snake_case",
    "description": "the situation, one clause",
    "terms": ["lowercase", "substrings", "that appear in such observations"],
    "expect": "help" | "no_help"}}
]

`terms` are matched case-insensitively as substrings against the observation
text; an observation matches the cue if ANY term is present. Choose terms that
are specific enough not to match everything. `expect` is your prior for the
direction only - it is checked against the measured rate, not trusted.
"""


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _shrunk_delta(k: int, m: int, pop: float, alpha: float) -> float:
    """Log-odds adjustment for a cue, shrunk toward the population rate.

    With little support this returns ~0, so a thin profile cannot move a
    decision and the score falls back to the situation prior. Cold start needs
    no special case: it is what this formula already does.
    """
    num = k + alpha * pop
    den = (m - k) + alpha * (1 - pop)
    return math.log(num / den) - _logit(pop)


@dataclass
class Rule:
    id: str
    description: str
    terms: list[str]
    k: int = 0
    m: int = 0
    delta_logodds: float = 0.0
    kappa: float = 0.0
    status: str = "active"

    @property
    def rate(self) -> float:
        return self.k / self.m if self.m else 0.0

    @property
    def tendency(self) -> str:
        return "wants_help" if self.delta_logodds > 0 else "does_not_want_help"

    def matches(self, inst: TaskInstance) -> bool:
        text = _describe(inst).lower()
        return any(t and t in text for t in self.terms)


@dataclass
class Profile:
    """The whole user model. Serialises to pipa/schemas/profile_v3.example.json."""
    user_id: str
    n_events: int
    population_rate: float
    user_rate: float
    user_positive: int
    beta_logodds: float
    kappa: float
    rules: list[Rule] = field(default_factory=list)
    retired: list[Rule] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        for r in d["rules"] + d["retired"]:
            r["rate"] = round(r["k"] / r["m"], 4) if r["m"] else None
        return d

    def score_adjustment(self, inst: TaskInstance) -> float:
        """Total log-odds the profile contributes for this observation."""
        return self.beta_logodds + sum(
            r.kappa * r.delta_logodds for r in self.rules if r.matches(inst))


_JSON = re.compile(r"\[.*\]", re.S)


def _parse_cues(text: str) -> list[dict]:
    m = _JSON.search(text or "")
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        terms = [str(t).lower().strip() for t in (c.get("terms") or [])
                 if str(t).strip()]
        if not terms:
            continue
        out.append({"id": str(c.get("id") or f"r{len(out)}")[:40],
                    "description": str(c.get("description") or "")[:120],
                    "terms": terms[:8]})
    return out


class RuleProfile:
    """PIPA v3: LLM-proposed cues, code-measured support."""

    select_history = staticmethod(_default_select)
    name = "b13_rules"

    def __init__(self, engine=None, taskset: TaskSet | None = None,
                 min_history: int = 8, max_rules: int = MAX_RULES,
                 min_support: int = MIN_SUPPORT):
        self.engine = engine
        self.taskset = taskset
        self.min_history = min_history
        self.max_rules = max_rules
        self.min_support = min_support
        self._cues: dict[tuple[str, int], list[dict]] = {}
        self._profiles: dict[tuple[str, int], Profile] = {}

    # -- population prior --------------------------------------------------
    @property
    def population_rate(self) -> float:
        """Base rate over the TRAIN split only, so test labels never inform it."""
        if not hasattr(self, "_pop"):
            labs = []
            if self.taskset is not None:
                labs = [i.target.get("label") for i in self.taskset.instances
                        if i.split == "train"]
            binary = [bool(x) for x in labs if isinstance(x, bool)]
            self._pop = sum(binary) / len(binary) if len(binary) >= 20 else 0.5
        return self._pop

    # -- construction ------------------------------------------------------
    def _propose(self, user_id: str, prefix: list[TaskInstance]) -> list[dict]:
        key = (user_id, len(prefix))
        if key in self._cues:
            return self._cues[key]
        cues: list[dict] = []
        if self.engine is not None:
            span = "unknown span"
            if prefix:
                days = (prefix[-1].t0 - prefix[0].t0).days
                span = f"{days} day span" if days else "under a day"
            prompt = TEMPLATE.format(n=len(prefix), span=span, k=self.max_rules,
                                     events=_render_events(prefix, seed=user_id))
            resp = self.engine.complete(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": prompt}],
                tag=f"cues:{user_id}:{len(prefix)}")
            if resp.ok:
                cues = _parse_cues(resp.text)
        self._cues[key] = cues
        return cues

    def _fit(self, user_id: str, prefix: list[TaskInstance],
             cues: list[dict]) -> Profile:
        """Measure every proposed cue against the WHOLE prefix.

        Counts do not depend on which events were shown to the model, which is
        why the rule list is stable across rebuilds even though cue proposal
        is sample-dependent.
        """
        pop = self.population_rate
        labelled = [h for h in prefix if isinstance(h.target.get("label"), bool)]
        pos = sum(bool(h.target["label"]) for h in labelled)
        n = len(labelled)

        beta = _shrunk_delta(pos, n, pop, ALPHA_USER) if n else 0.0
        prof = Profile(user_id=user_id, n_events=len(prefix),
                       population_rate=round(pop, 4),
                       user_rate=round(pos / n, 4) if n else 0.0,
                       user_positive=pos, beta_logodds=round(beta, 4),
                       kappa=round(n / (n + ALPHA_USER), 4) if n else 0.0)

        for c in cues:
            r = Rule(id=c["id"], description=c["description"], terms=c["terms"])
            hits = [h for h in labelled if r.matches(h)]
            r.m = len(hits)
            r.k = sum(bool(h.target["label"]) for h in hits)
            if r.m < self.min_support:
                r.status = "retired"
                r.kappa = 0.0
                prof.retired.append(r)
                continue
            r.delta_logodds = round(_shrunk_delta(r.k, r.m, pop, ALPHA_RULE), 4)
            r.kappa = round(r.m / (r.m + ALPHA_RULE), 4)
            prof.rules.append(r)

        # bounded: keep the rules that move the decision most
        prof.rules.sort(key=lambda r: -abs(r.kappa * r.delta_logodds))
        prof.retired.extend(prof.rules[self.max_rules:])
        prof.rules = prof.rules[: self.max_rules]
        return prof

    def profile_for(self, user_id: str, prefix: list[TaskInstance]) -> Profile:
        """Cached per (owner, bucket), matching `ProfileReconstructor`.

        `user_id` is the owner of the HISTORY, not the person being predicted.
        A donor control calls `build` with the target's instance and the
        donor's history; keying on the target would make the control collide
        with the real profile and silently serve whichever was built first,
        destroying the one control the project rests on - correction 6 again.

        Note `history_ids` is a sliding 512-event window rather than a growing
        prefix, so "the first N events" shifts between instances. Keying on the
        bucket freezes the profile at its first build for that bucket, which is
        what b5 already does; content-based keys cost one LLM call per instance.
        """
        key = (user_id, len(prefix))
        if key not in self._profiles:
            self._profiles[key] = self._fit(user_id, prefix,
                                            self._propose(user_id, prefix))
        return self._profiles[key]

    # -- rendering ---------------------------------------------------------
    def _render(self, prof: Profile, instance: TaskInstance) -> str:
        rel = prof.user_rate - prof.population_rate
        direction = ("more" if rel > 0.02 else
                     "less" if rel < -0.02 else "about as")
        lines = [
            f"BASELINE: wants help in {prof.user_rate:.0%} of past situations "
            f"({prof.user_positive} of {prof.n_events}) against {prof.population_rate:.0%} "
            f"for a typical user - {direction} interruptible than average.",
            "",
            f"WHEN HELP IS WANTED (measured over all {prof.n_events} past events):",
        ]
        if not prof.rules:
            lines.append("- no situation has been seen often enough yet; "
                         "judge from the current observation alone.")
        for r in prof.rules:
            mark = ">> NOW: " if r.matches(instance) else "   "
            lines.append(f"{mark}{r.description} - help wanted {r.k}/{r.m} "
                         f"({r.rate:.0%})")
        return "\n".join(lines)[:MAX_CARD_CHARS]

    # The rule form conditions a BINARY outcome on a situation, so it applies
    # only to tasks whose label is a bool. On `intention` and `suggestion` the
    # label is a string: `_fit` then finds no labelled events and returns an
    # empty card, making this condition a silent duplicate of b0_none. That
    # cost two void conditions across two tasks before it was noticed, so the
    # unsupported case now raises instead of degrading quietly.
    SUPPORTED = ("timing_annotated", "timing_production")

    def build(self, instance: TaskInstance,
              history: list[TaskInstance]) -> ProfileCard:
        if instance.task_type not in self.SUPPORTED:
            raise ValueError(
                f"{self.name} supports {self.SUPPORTED}, not "
                f"{instance.task_type!r}: rules condition a binary outcome and "
                f"this task's label is not a bool. Exclude this condition, or "
                f"extend Rule to carry a class distribution.")
        n = _bucket(len(history))
        if n < self.min_history:
            return ProfileCard("", self.name, len(history),
                               {"reason": "insufficient history", "bucket": n})
        prefix = history[:n]
        # the profile belongs to whoever the history belongs to, not to whoever
        # is being predicted - so donor controls export honestly
        owner = prefix[0].user_id if prefix else instance.user_id
        prof = self.profile_for(owner, prefix)
        if not prof.rules and prof.kappa == 0.0:
            return ProfileCard("", self.name, len(history), {"bucket": n})
        # Full rule detail is logged per decision so ablations (drop a rule,
        # drop beta, drop kappa weighting, re-score without the LLM) can be run
        # offline from stored predictions without any new API calls.
        return ProfileCard(
            self._render(prof, instance), self.name, len(history),
            {"bucket": n, "built_from": prof.n_events, "owner": prof.user_id,
             "n_rules": len(prof.rules), "n_retired": len(prof.retired),
             "population_rate": prof.population_rate,
             "user_rate": prof.user_rate, "user_positive": prof.user_positive,
             "beta_logodds": prof.beta_logodds, "beta_kappa": prof.kappa,
             "adjustment": round(prof.score_adjustment(instance), 4),
             "matched": [r.id for r in prof.rules if r.matches(instance)],
             "rules": [{"id": r.id, "k": r.k, "m": r.m,
                        "delta": r.delta_logodds, "kappa": r.kappa,
                        "hit": r.matches(instance), "terms": r.terms}
                       for r in prof.rules]})

    # -- diagnostics -------------------------------------------------------
    @property
    def n_profiles_built(self) -> int:
        return len(self._profiles)

    def built_profiles(self) -> dict[tuple[str, int], str]:
        return {k: json.dumps(v.to_json(), ensure_ascii=False)
                for k, v in self._profiles.items()}


def rule_churn(a: Profile, b: Profile) -> float:
    """Jaccard distance over rule identities between consecutive rebuilds.

    Replaces the character-trigram Consistency Score with something readable:
    held / revised / retired, rather than 0.41.
    """
    ida, idb = {r.id for r in a.rules}, {r.id for r in b.rules}
    if not (ida or idb):
        return 0.0
    return 1 - len(ida & idb) / len(ida | idb)
