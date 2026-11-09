"""The posterior layer: a Beta-Binomial belief over each user's receptivity.

    from bayes import BetaPosterior, Costs, fit_prior, decide, decide_mean, decide_oracle

This is the piece the Method section promises and the code did not have. The
existing `rules._shrunk_delta` is the posterior MEAN of exactly this model,
collapsed to a scalar - so everything downstream of it was already "PIPA-Mean".
This module keeps the whole distribution.

WHY BETA-BINOMIAL
The outcome is binary (did the user want help), so receptivity theta_u is a
rate and Beta is its conjugate prior. Posterior, mean, variance and credible
intervals are closed-form, so nothing is approximate, and "7/10 vs 70/100" is
literally Beta(a+7,b+3) vs Beta(a+70,b+30): near-identical means, an order of
magnitude apart in variance.

THE DECISION PROBLEM, STATED HONESTLY
Three actions: act, ask, silent. Costs are asymmetric and fixed in advance.
Under a utility that is LINEAR in theta, expected utility is

    E_theta[U(a, theta)] = U(a, E[theta])

so the posterior MEAN is a sufficient statistic and the Bayes-optimal policy
needs nothing else. This is a real constraint and the paper should say it:
a distribution cannot beat its own mean in expectation under linear utility.

Where the spread DOES matter:

  (i)  a risk-averse policy that acts only when posterior mass is confidently
       past the threshold - it gives up some expected utility to avoid
       expensive false alarms when the evidence is thin;
  (ii) knowing WHEN to ask: two users with the same mean but different
       evidence get different actions, which a point estimate cannot do.

The experiments measure exactly these two things and nothing more.

POLICIES
  oracle     knows theta*, picks argmax_a U(a, theta*)          upper bound
  PIPA-Mean  picks argmax_a U(a, E[theta|H])                    Bayes-optimal
                                                                 under linear U
  PIPA       act    if P(theta > tau_act | H)    >= 1 - eps
             silent if P(theta < tau_silent | H) >= 1 - eps
             ask    otherwise                                    risk-averse

tau_act and tau_silent are DERIVED from the costs (the theta at which act
overtakes ask, and ask overtakes silent), not chosen by hand, so every arm
faces the same decision boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ACTIONS = ("silent", "ask", "act")


def fit_prior(rates: np.ndarray, counts: np.ndarray, floor: float = 0.5) -> tuple[float, float]:
    """Empirical-Bayes Beta(a0, b0) from per-user rates, method of moments,
    count-weighted so noisy few-observation rates pull less."""
    w = counts / counts.sum()
    m = float((w * rates).sum())
    v = max(float((w * (rates - m) ** 2).sum()), 1e-4)
    common = m * (1 - m) / v - 1
    return max(m * common, floor), max((1 - m) * common, floor)


@dataclass(frozen=True)
class BetaPosterior:
    a: float
    b: float

    @classmethod
    def prior(cls, a0: float, b0: float) -> "BetaPosterior":
        return cls(a0, b0)

    def update(self, k: int, m: int) -> "BetaPosterior":
        """k positives out of m observations."""
        return BetaPosterior(self.a + k, self.b + (m - k))

    def forget(self, gamma: float) -> "BetaPosterior":
        """Discount accumulated evidence toward the prior scale - the response
        to drift. gamma=1 keeps everything; gamma<1 widens the interval."""
        return BetaPosterior(self.a * gamma, self.b * gamma)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1))

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1 - level) / 2
        return (float(stats.beta.ppf(lo, self.a, self.b)),
                float(stats.beta.ppf(1 - lo, self.a, self.b)))

    def prob_above(self, tau: float) -> float:
        return float(stats.beta.sf(tau, self.a, self.b))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.beta(self.a, self.b, size=n)


@dataclass(frozen=True)
class Costs:
    """U(action, wanted). Each action must be optimal on SOME theta range, or
    the three-way choice is vacuous. With these defaults:
        act    2*theta - 1        best for theta > 0.583
        ask    0.8*theta - 0.3    best for 0.231 < theta < 0.583
        silent -0.5*theta         best for theta < 0.231
    """
    act_wanted: float = 1.0
    act_unwanted: float = -1.0
    ask_wanted: float = 0.5       # help arrives, with friction and delay
    ask_unwanted: float = -0.3    # an interruption, milder than a wrong act
    silent_wanted: float = -0.5
    silent_unwanted: float = 0.0

    def utility(self, action: str, wanted: bool) -> float:
        return {
            "act": (self.act_wanted, self.act_unwanted),
            "ask": (self.ask_wanted, self.ask_unwanted),
            "silent": (self.silent_wanted, self.silent_unwanted),
        }[action][0 if wanted else 1]

    def expected(self, action: str, theta: float) -> float:
        return theta * self.utility(action, True) + (1 - theta) * self.utility(action, False)

    def thresholds(self) -> tuple[float, float]:
        """(tau_silent, tau_act): where ask overtakes silent, and act overtakes ask."""
        def cross(a1, a2):
            # theta at which expected(a1) == expected(a2)
            d1 = self.utility(a1, True) - self.utility(a1, False)
            d2 = self.utility(a2, True) - self.utility(a2, False)
            c1, c2 = self.utility(a1, False), self.utility(a2, False)
            return (c2 - c1) / (d1 - d2)
        return cross("ask", "silent"), cross("act", "ask")

    def best(self, theta: float) -> str:
        return max(ACTIONS, key=lambda a: self.expected(a, theta))


def decide_oracle(theta_star: float, costs: Costs) -> str:
    return costs.best(theta_star)


def decide_mean(post: BetaPosterior, costs: Costs) -> str:
    """PIPA-Mean: plug E[theta] into expected utility. Bayes-optimal under
    linear utility - this is the strong ablation, not a straw man."""
    return costs.best(post.mean)


def decide(post: BetaPosterior, costs: Costs, eps: float = 0.2) -> str:
    """PIPA: risk-averse, decides from posterior MASS past the cost-derived
    thresholds. Asks when neither tail is confident."""
    tau_silent, tau_act = costs.thresholds()
    if post.prob_above(tau_act) >= 1 - eps:
        return "act"
    if 1 - post.prob_above(tau_silent) >= 1 - eps:
        return "silent"
    return "ask"
