# Cost Log — Tenacious-Bench v0.1

Per-bucket spend record. Updated after every API call and any compute charge. Source of truth: `cost/openrouter_calls.jsonl` (machine-readable per-call log).

## Budget envelope (Week 11)

| Bucket | Budget | Pays for |
|---|---:|---|
| Dataset authoring (dev-tier LLM, synthesis + judge) | $3–5 | OpenRouter, Days 2–3 |
| Training | $0–5 | Free on Colab T4 (default); RunPod fallback only if Colab caps |
| Held-out evaluation (eval-tier model, sealed slice only) | $2–3 | Day 6 ablations |
| Reserve | $1–2 | Bug fixes, re-runs |
| **Total envelope** | **$10** | (program raised from $20 on 2026-04-23) |

## Spend to date — interim (Acts I + II)

| Date | Bucket | Description | Calls | Tokens (in/out) | Cost (USD) |
|---|---|---|---:|---|---:|
| 2026-04-29 | Authoring | OpenRouter smoke test (Qwen3-Next, 1 call) | 1 | 35 / 4 | $0.0000 |
| 2026-04-29 | Authoring | Synthesis dry run (DeepSeek + Qwen, 4 tasks × 2 calls) | 8 | ~3,200 / ~800 | $0.0008 |
| 2026-04-29 | Authoring | Synthesis full run (16 seeds × 4 variants × 2 calls = 128, 2 timeouts) | 126 | ~50,000 / ~12,000 | $0.0079 |
| 2026-04-30 | Authoring | Judge calibration (Llama-3.3-70B, 50 templated tasks; 1 SSL timeout) | 49 | ~17,000 / ~3,000 | $0.0027 |
| | | **Subtotal — interim authoring spend** | **184** | | **$0.0114** |
| 2026-04-30 | Training | (not started — Path B training is Day 4–5) | 0 | | $0.0000 |
| 2026-04-30 | Training | Training data conversion (128 preference pairs, local compute) | 1 | ~0 / ~0 | $0.0000 |
| 2026-04-30 | Held-out eval | (not started — Day 6) | 0 | | $0.0000 |

## Headroom remaining

- Authoring envelope: $0.0114 of $5.00 spent → **$4.99 left** for any v0.1.1 re-runs.
- Training envelope: $0 of $5 spent → unchanged.
- Held-out eval envelope: $0 of $3 spent → unchanged.
- Total envelope: $0.0114 of $10 → **$9.99 left**.

## Cost-discipline notes (per the brief's non-negotiable rules)

1. **No τ²-Bench retail re-runs.** Confirmed — the Week 10 score is reused; no τ²-retail spend in this week's log.
2. **No eval-tier model on Days 2–3.** Confirmed — all interim authoring spend is on dev-tier models (DeepSeek-V3.2, Qwen3-Next-80B, Llama-3.3-70B). Claude Sonnet 4.6 / GPT-5 class are *not* called until Day 6.
3. **Per-call cost log.** Confirmed — `cost/openrouter_calls.jsonl` records timestamp, model, purpose, prompt/completion tokens, and cost-USD per call.

## Reproduction

```bash
# Tally current spend
python -c "
import json
total=0; n=0
for line in open('cost/openrouter_calls.jsonl', encoding='utf-8'):
    r = json.loads(line); total += r.get('cost_usd', 0); n += 1
print(f'{n} calls, total cost: ${total:.4f}')
"
```
