"""Inference engine: one call path, cached, budgeted, and dry-runnable.

Design points that matter for a paper:

* **Dry run is the default.** Nothing reaches the network unless you pass
  --live. You can inspect every rendered prompt before spending anything.
* **Disk cache** keyed on (model, messages, params). Re-running an experiment
  after a crash costs nothing, and conditions that produce identical prompts
  are charged once.
* **Hard budget.** The runner aborts when the estimated spend cap is hit
  instead of quietly draining an account.
* Every call records tokens and latency, so E6's efficiency numbers come from
  the same path that produced the predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CACHE_DB = ROOT / "runs" / "llm_cache.sqlite"

# Prices are USD per 1M tokens; adjust to your account's actual rates.
MODELS: dict[str, dict] = {
    "qwen-plus": {
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 0.4, "price_out": 1.2,
    },
    "qwen-max": {
        "model_name": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 1.6, "price_out": 6.4,
    },
    "qwen-turbo": {
        "model_name": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 0.05, "price_out": 0.2,
    },
    # DeepSeek served through DashScope, so one key covers both families.
    # deepseek-v4-flash is what the production deployment actually routed to
    # (see llm_usage_records.model_alias), which keeps b6_deployed on a fair
    # footing. Prices below are ESTIMATES - verify against your billing page;
    # they only drive the budget guard, so they are set conservatively high.
    "deepseek-v4-flash": {
        "model_name": "deepseek-v4-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 0.15, "price_out": 0.60,
    },
    "deepseek-v3.2": {
        "model_name": "deepseek-v3.2",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 0.28, "price_out": 1.10,
    },
    "deepseek-v4-pro": {
        "model_name": "deepseek-v4-pro",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "price_in": 0.60, "price_out": 2.40,
    },
    "deepseek-chat": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "price_in": 0.27, "price_out": 1.1,
    },
    # Third backbone. Chosen because it is a different vendor family from
    # qwen/deepseek, so the cross-backbone table stops being two variants of
    # the same lineage. Earlier runs went through the Bianxie proxy; that
    # account is out of credit, so this now goes to OpenAI directly. Same
    # model id, same list price - only the route changed. Runs record the
    # base_url in their summary so the provider is always traceable.
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "price_in": 0.15, "price_out": 0.60,
    },
    # local vLLM / Ollama with an OpenAI-compatible endpoint
    "local": {
        "model_name": os.environ.get("PIPA_LOCAL_MODEL", "qwen2.5-7b-instruct"),
        "base_url": os.environ.get("PIPA_LOCAL_BASE_URL", "http://localhost:8000/v1"),
        "env_key": "PIPA_LOCAL_API_KEY",
        "price_in": 0.0, "price_out": 0.0,
    },
}


def load_env() -> None:
    """Read API keys from a .env file without overwriting real env vars.

    Looked for in order: $PIPA_ENV, then <repo>/.env. Copy .env.example to
    .env and fill in the providers you intend to use; a key is only needed
    for the provider whose model you pass to --model.
    """
    candidates = []
    if os.environ.get("PIPA_ENV"):
        candidates.append(Path(os.environ["PIPA_ENV"]))
    candidates.append(PROJECT / ".env")
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        import statistics as st
        return {
            "calls": self.calls, "cached": self.cached, "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "median_latency_ms": round(st.median(self.latency_ms), 1)
            if self.latency_ms else 0.0,
        }


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "k TEXT PRIMARY KEY, text TEXT, pt INTEGER, ct INTEGER)")
        self._con.commit()

    def get(self, key: str):
        with self._lock:
            row = self._con.execute(
                "SELECT text, pt, ct FROM cache WHERE k=?", (key,)).fetchone()
        return row

    def put(self, key: str, text: str, pt: int, ct: int) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (key, text, pt, ct))
            self._con.commit()


class ProviderOutOfCredit(RuntimeError):
    """The provider refused the call for lack of funds (e.g. HTTP 402).

    Distinct from BudgetExceeded, which is our own spend cap. This one means
    no further call can succeed until a human tops up the account, so the run
    must stop rather than fill the output with error rows.
    """


def _is_out_of_credit(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("insufficient balance" in s or "402" in s
            or "insufficient_quota" in s or "exceeded your current quota" in s)


class BudgetExceeded(RuntimeError):
    pass


class LLMEngine:
    """Thread-safe for concurrent use: usage counters are mutated under a
    lock and the sqlite cache has its own."""

    def __init__(self, model: str = "qwen-plus", *, dry_run: bool = True,
                 temperature: float = 0.0, max_tokens: int = 800,
                 budget_usd: float = 5.0, cache: bool = True,
                 max_retries: int = 4, timeout: float = 90.0):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; known: {list(MODELS)}")
        self.model_key = model
        self.cfg = MODELS[model]
        self.dry_run = dry_run
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.budget_usd = budget_usd
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage = Usage()
        self._cache = _Cache(CACHE_DB) if cache else None
        self._client = None
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self.dry_run_prompts: list[dict] = []

    def _record(self, *, pt=0, ct=0, latency=None, cached=False, error=False,
                cost=0.0) -> None:
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.cached += cached
            self.usage.errors += error
            self.usage.prompt_tokens += pt
            self.usage.completion_tokens += ct
            self.usage.cost_usd += cost
            if latency is not None:
                self.usage.latency_ms.append(latency)

    # -- client ------------------------------------------------------------
    def _get_client(self):
        with self._client_lock:
            return self._build_client()

    def _build_client(self):
        if self._client is None:
            load_env()
            key = os.environ.get(self.cfg["env_key"])
            if not key:
                raise RuntimeError(
                    f"{self.cfg['env_key']} not set. Add it to the repo's .env "
                    f"or export it, or run with dry_run=True.")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=self.cfg["base_url"],
                                  timeout=self.timeout)
        return self._client

    # -- cost --------------------------------------------------------------
    def _cost(self, pt: int, ct: int) -> float:
        return (pt * self.cfg["price_in"] + ct * self.cfg["price_out"]) / 1e6

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps({
            "m": self.cfg["model_name"], "b": self.cfg["base_url"],
            "t": self.temperature, "mt": self.max_tokens, "msg": messages,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- main --------------------------------------------------------------
    def complete(self, messages: list[dict], *, tag: str = "") -> LLMResponse:
        key = self._key(messages)
        if self._cache:
            row = self._cache.get(key)
            if row is not None:
                self._record(cached=True)
                return LLMResponse(row[0], row[1], row[2], 0.0, cached=True)

        if self.dry_run:
            self.dry_run_prompts.append({"tag": tag, "messages": messages})
            approx = sum(len(m["content"]) for m in messages) // 3
            self._record(pt=approx, cost=self._cost(approx, self.max_tokens // 2))
            return LLMResponse("", approx, 0, 0.0, error="dry_run")

        if self.usage.cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.usage.cost_usd:.2f} >= budget ${self.budget_usd:.2f}")

        client = self._get_client()
        delay = 1.5
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=self.cfg["model_name"],
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                dt = (time.time() - t0) * 1000
                text = (resp.choices[0].message.content or "").strip()
                u = getattr(resp, "usage", None)
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0
                self._record(pt=pt, ct=ct, latency=dt, cost=self._cost(pt, ct))
                if self._cache and text:
                    self._cache.put(key, text, pt, ct)
                return LLMResponse(text, pt, ct, dt)
            except Exception as exc:  # noqa: BLE001 - surface any provider error
                last_err = f"{type(exc).__name__}: {exc}"
                # A funding failure is not transient. Retrying for a few seconds
                # cannot fix it, and continuing writes thousands of unusable
                # rows that look like data - this happened twice, costing a run
                # of 2,458 rows each time. Abort the whole run instead.
                if _is_out_of_credit(exc):
                    self._record(error=True)
                    raise ProviderOutOfCredit(
                        "provider rejected the request for lack of credit: "
                        f"{last_err}. Top up the account, then re-run; nothing "
                        "usable is produced while this persists.") from exc
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        self._record(error=True)
        return LLMResponse("", 0, 0, 0.0, error=last_err)

    def check_connection(self) -> LLMResponse:
        """One tiny live call, to verify credentials before a long run."""
        was_dry = self.dry_run
        self.dry_run = False
        try:
            return self.complete([{"role": "user", "content": "Reply with: OK"}],
                                 tag="healthcheck")
        finally:
            self.dry_run = was_dry
