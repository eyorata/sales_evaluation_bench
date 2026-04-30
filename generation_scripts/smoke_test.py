"""One cheap call to confirm OPENROUTER_API_KEY works and cost-log writes."""
from __future__ import annotations

from . import openrouter_client


def main() -> int:
    out = openrouter_client.chat(
        model="qwen/qwen3-next-80b-a3b-instruct",
        system="You are a careful tester. Reply with one short sentence.",
        user="Reply with exactly: 'ok smoke 1'.",
        temperature=0.0,
        max_tokens=16,
        purpose="smoke_test",
    )
    print("response:", out["text"][:120])
    print("cost_usd:", out["cost_usd"])
    print("usage:", out["usage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
