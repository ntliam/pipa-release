"""Shared helpers for the PIPA dataset build."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

# --------------------------------------------------------------------------
# streaming readers
# --------------------------------------------------------------------------

_DECODER = json.JSONDecoder()


def stream_json_array(path: str | Path, chunk_size: int = 1 << 22) -> Iterator[Any]:
    """Yield elements of a top-level JSON array without loading the whole file.

    full_dataset.json is ~440 MB; json.load() on it costs several GB of RAM.
    """
    buf = ""
    started = False
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.lstrip()
                if not started:
                    if not s:
                        buf = s
                        break
                    if s[0] != "[":
                        raise ValueError(f"{path} does not start with a JSON array")
                    s = s[1:].lstrip()
                    started = True
                if s.startswith(","):
                    s = s[1:].lstrip()
                if not s or s.startswith("]"):
                    buf = s
                    break
                try:
                    obj, end = _DECODER.raw_decode(s)
                except ValueError:
                    buf = s
                    break  # need more input
                buf = s[end:]
                yield obj


def stream_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield records from a .jsonl file, skipping unparseable lines."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

SALT = "pipa-v1"


def short_hash(value: str, prefix: str = "", n: int = 12) -> str:
    digest = hashlib.sha256((SALT + str(value)).encode("utf-8")).hexdigest()[:n]
    return f"{prefix}{digest}"


# --------------------------------------------------------------------------
# text hygiene
# --------------------------------------------------------------------------

# Window titles in full_dataset.json are padded with runs of zero-width
# characters (​‌‍⁠﻿), which break tokenisation.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_WS = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _INVISIBLE.sub("", text)
    return _WS.sub(" ", text).strip()


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# leakage lexicon
# --------------------------------------------------------------------------

# Names of assistant products and explicit "asking an AI" phrasings. A hit in
# the *observable* context means the screen already shows an assistant, which
# would let a model shortcut the need_help decision. We flag, never delete, so
# the residual leak stays measurable.
LEAK_TERMS = [
    "chatgpt", "openai", "deepseek", "kimi", "claude", "copilot", "gemini",
    "perplexity", "chatglm", "grok",
    "通义", "千问", "qwen", "豆包", "文心一言", "讯飞星火", "智谱", "元宝",
    "秘塔", "腾讯元宝", "纳米ai", "天工",
    "向ai", "问ai", "ai助手", "ai 助手", "人工智能助手", "智能助手",
    "大模型", "对话助手", "提问框",
]
_LEAK_RE = re.compile("|".join(re.escape(t) for t in LEAK_TERMS), re.IGNORECASE)


def leak_terms_in(text: str) -> list[str]:
    if not text:
        return []
    return sorted({m.group(0).lower() for m in _LEAK_RE.finditer(text)})


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
