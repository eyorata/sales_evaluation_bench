"""Thin OpenRouter client used by the synthesis and judge scripts.

Reads OPENROUTER_API_KEY from environment (loaded by python-dotenv if installed).
No retries beyond 1; failures bubble up so the caller can decide whether to fall
back to the offline template path.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COST_LOG = REPO_ROOT / "cost" / "openrouter_calls.jsonl"


def _load_env() -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


_load_env()


def _log(call: dict[str, Any]) -> None:
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(call, ensure_ascii=False) + "\n")


# OpenRouter dev-tier per-token costs (USD per 1M tokens). These are coarse
# estimates used for the cost log; OpenRouter's response carries authoritative
# usage stats and we record those when present.
COST_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v3.2-exp": {"prompt": 0.14, "completion": 0.28},
    "deepseek/deepseek-chat-v3.1": {"prompt": 0.14, "completion": 0.28},
    "qwen/qwen3-next-80b-a3b-instruct": {"prompt": 0.14, "completion": 0.56},
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.13, "completion": 0.40},
}


def chat(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.4,
    max_tokens: int = 800,
    timeout: int = 60,
    purpose: str = "unspecified",
) -> dict[str, Any]:
    """Single chat completion. Returns {'text', 'usage', 'cost_usd', 'raw'}."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tenacious-bench",
            "X-Title": "Tenacious-Bench v0.1",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        _log({"model": model, "purpose": purpose, "error": str(e), "body": body[:500], "wall_ms": int((time.time() - t0) * 1000)})
        raise

    obj = json.loads(body)
    text = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage") or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    rates = COST_PER_M_TOKENS.get(model, {"prompt": 0.20, "completion": 0.40})
    cost_usd = (pt * rates["prompt"] + ct * rates["completion"]) / 1_000_000
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "purpose": purpose,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd": round(cost_usd, 6),
        "wall_ms": int((time.time() - t0) * 1000),
    }
    _log(rec)
    return {"text": text, "usage": usage, "cost_usd": cost_usd, "raw": obj}
