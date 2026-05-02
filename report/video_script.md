# Demo Video — 6-Minute Walkthrough Script

**Total cap:** 6:00 (360s). Hard cap; the rubric explicitly checks for time-cap compliance.

**Rubric weights** drive the time budget: dataset (25) + task scoring (25) + ablation (25) get the heaviest allocation; blog + community (15) is lighter; framing and recap (10) is minimal.

| Segment | Time | Rubric pts |
|---|---:|---:|
| 0. Intro (15 s) | 0:00 – 0:15 | (recap counts toward time-discipline) |
| 1. Dataset walkthrough on HuggingFace | 0:15 – 1:35 (80 s) | 25 |
| 2. End-to-end task scoring | 1:35 – 2:50 (75 s) | 25 |
| 3. Ablation result with traceability | 2:50 – 4:30 (100 s) | 25 |
| 4. Blog post (dev.to) | 4:30 – 5:00 (30 s) | 15 (combined) |
| 5. Community engagement (τ²-Bench issue) | 5:00 – 5:30 (30 s) | 15 (combined) |
| 6. Wrap | 5:30 – 6:00 (30 s) | (time-discipline) |

---

## Pre-recording checklist (do this first; ~5 minutes)

### Tabs to have open in your browser, in this order
1. **HuggingFace dataset:** https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1
2. **Blog post:** https://dev.to/eyorata/when-your-training-loss-is-lying-to-you-building-a-tenacious-specific-sales-outreach-benchmark-2jgd
3. **Community-engagement issue:** https://github.com/sierra-research/tau2-bench/issues/295
4. **GitHub repo (optional fallback):** https://github.com/eyorata/sales_evaluation_bench

### Files to have open in VS Code, in this order (split-screen so you can switch fast)
1. `tenacious_bench_v0.1/dev/tasks.jsonl` — open and **navigate to a TB-0204 row** (style-guide pair SG-01, has ground_truth.chosen + rejected). Use Ctrl+F → `TB-0700` if `dev` doesn't have one with ground truth; held_out has the SG pairs.
2. `ablations/ablation_results.json` — pre-scrolled to the `delta_a` block.
3. `ablations/held_out_traces.jsonl` — pre-scrolled to `pair_idx: 0`.
4. `scoring_evaluator.py` — open at the top so you can show a check function if asked.

### Terminal ready in repo root
```bash
cd "c:/Users/user/Documents/tenx_academy/sales_evaluation_bench"
# Pre-test the command so it runs cleanly during recording:
python scoring_evaluator.py --self-test
```
You should see `6/6 PASS`. If anything errors, fix BEFORE recording.

### Recording tool (no-login output requirement)
Brief says "no login" for viewer access:
- **Best:** OBS Studio (free) → MP4 → upload to YouTube **Unlisted** or Google Drive set to "anyone with link". Both let viewers watch without signing in.
- **Quick (Windows):** Win+G → Xbox Game Bar → record screen → MP4 → upload as above.
- **Avoid:** Loom default (asks viewer to sign up), Vimeo private (asks for password).

### Mic test
Record 10 s of yourself counting "1, 2, 3" and play it back. If it sounds muffled, either move the mic closer or record a fresh take. Audio intelligibility is one of the rubric items.

---

## Narration script (verbatim — read this on-screen as a teleprompter)

> **Tip:** italicize/whisper the parenthetical "*pause / scroll*" cues to yourself. The voice should be conversational, not read-aloud. Practice the segment timing once before recording for real.

### Segment 0 — Intro (0:00 – 0:15) [15 s]

**[on screen: VS Code with `README.md` open, or the HuggingFace tab not yet selected]**

> "Hi, I'm Eyoel Nebiyu. This is Tenacious-Bench v0.1 — a 266-task evaluation benchmark for B2B sales-outreach agents that grades ten Tenacious-specific failure modes that public benchmarks like τ²-retail don't cover. In the next six minutes I'll walk you through the dataset on HuggingFace, score one task end-to-end, show the ablation result, and link the public artifacts."

**[switch to HuggingFace tab]**

---

### Segment 1 — Dataset walkthrough on HuggingFace (0:15 – 1:35) [80 s]

**Rubric checklist for this segment:**
- [ ] Live URL in address bar (not screenshot)
- [ ] Datasheet scrolled showing all 7 Gebru sections
- [ ] All 3 partitions named with task counts
- [ ] Source-mode metadata visible on at least one task
- [ ] License visible
- [ ] Narration throughout

**[on screen: HuggingFace dataset page, address bar visible, browser zoomed to ~110%]**

> "[0:15] This is the HuggingFace dataset page — `huggingface.co/datasets/eyorata/tenacious_bench_v0.1` — license is **CC-BY-4.0**, you can see that in the sidebar."

**[scroll to the Files tab]**

> "[0:30] The dataset has three partitions: **train with 128 tasks, dev with 63 tasks, and held-out with 75 tasks** — held-out is sealed at publish, so what's visible is `SEALED.md` explaining how to request access. The other two are downloadable JSONL."

**[click on `dev/tasks.jsonl`, scroll to a row with full input + metadata]**

> "[0:50] Each task carries `source_mode` — here it's `style_guide_pair` — plus `dimension`, `difficulty`, the full input, the rubric weights, and `ground_truth.chosen_output` and `rejected_output` for the preference-pair tasks."

**[click back to dataset root, click Datasheet/README link, scroll smoothly through the 7 sections]**

> "[1:10] The datasheet follows Gebru's seven sections — **motivation, composition, collection, preprocessing, uses, distribution, maintenance** — plus Pushkarna's telescopic, periscopic, microscopic layered detail. Source modes, contamination checks, and inter-rater agreement are all here."

**[switch to next tab — task scoring]**

---

### Segment 2 — End-to-end task scoring (1:35 – 2:50) [75 s]

**Rubric checklist:**
- [ ] Task input fields visible
- [ ] Candidate output visible
- [ ] Evaluator producing score on screen
- [ ] Score broken down by rubric dimension
- [ ] One specific rubric check applied to this output
- [ ] Narration connecting all four

**[on screen: VS Code with `tenacious_bench_v0.1/held_out/tasks.jsonl` open, scrolled to a task with ground_truth — try TB-0700 (SG-01 — Series A funding)]**

> "[1:35] Picking one task end-to-end. This is **TB-0700**, a held-out style-guide pair. The scenario is on screen: a Series A prospect at $14M raised in February with Python role count rising from 2 to 7 in 60 days. The rubric has four checks — `no_banned_phrases`, `policy_compliant`, `length`, and `tone_marker_judge`."

**[scroll right or wrap-show the chosen output]**

> "[2:00] Here's `ground_truth.chosen_output` — the Maya draft that grounds on the funding and role velocity. And the `rejected_output` is BAD-#1, the world-class talent / top-talent self-promotion wall."

**[switch to terminal — run a one-liner that scores both]**

```bash
python -c "
import json
from scoring_evaluator import score_task
t = next(json.loads(l) for l in open('tenacious_bench_v0.1/held_out/tasks.jsonl', encoding='utf-8') if '\"task_id\": \"TB-0700\"' in l)
chosen = t['ground_truth']['chosen_output']
rejected = t['ground_truth']['rejected_output']
print('CHOSEN  →', score_task(t, chosen))
print('REJECTED →', score_task(t, rejected))
"
```

**[as the output prints]**

> "[2:25] The chosen scores around 1.0 — passes every check. The rejected scores around 0.3 — `no_banned_phrases` fires on **'world-class', 'top talent', 'gold standard'**, exactly the failure mode we encoded. The `tone_marker_judge` check returns the dimension breakdown — direct, grounded, honest, professional, non-condescending."

> "[2:45] No human in the loop — that's the whole point of mechanically gradable rubrics."

**[switch to next file — ablation]**

---

### Segment 3 — Ablation result with traceability (2:50 – 4:30) [100 s]

**Rubric checklist:**
- [ ] Ablation numeric value on screen
- [ ] CI or p-value visible
- [ ] Held-out trace artifact opened
- [ ] One numeric claim traced to a specific source row
- [ ] Narration walks through the inferential chain
- [ ] Negative/null deltas treated equivalently

**[on screen: `ablations/ablation_results.json` open in VS Code]**

> "[2:50] This is `ablations/ablation_results.json`. The headline numbers — Delta A is plus 25 percentage points: trained accuracy 0.417, untrained baseline 0.167, **paired bootstrap p equals 0.0316**, 95% CI 0 to 0.5. So — directionally positive at p less than 0.05."

**[scroll down to delta_b block]**

> "[3:15] Delta B is the honest negative — minus 42 percentage points. Trained at 0.417 vs prompt-engineered same-backbone at 0.833, p equals 0.99. The prompt baseline beats the trained adapter decisively. **The brief explicitly names a negative Delta B as a publishable finding** — that's the lesson of the project."

**[open `ablations/held_out_traces.jsonl`]**

> "[3:35] Now I'll trace one number back. This is `held_out_traces.jsonl` — one row per held-out pair. Pair index 0:"

**[show the first row]**

```json
{"pair_idx": 0,
 "trained":   {"chosen_lp": -4.30, "rejected_lp": -2.90, "prefers_chosen": 0},
 "untrained": {"chosen_lp": -3.49, "rejected_lp": -2.35, "prefers_chosen": 0},
 "prompt_engineered": {"pick": "B", "swap": true, "prefers_chosen": 1}}
```

> "[3:55] On this pair, both the trained and untrained models prefer the rejected — they get this one wrong. The prompt-engineered judge gets it right. Multiply this across all 12 held-out pairs and you get the trained 5/12, untrained 2/12, prompt 10/12 numbers from the ablation table."

**[scroll back to `ablation_results.json` to show 'verdict']**

> "[4:15] Verdict on Delta A is `POSITIVE_NOT_SIGNIFICANT` because the CI lower bound grazes zero — that's a small-n artifact at n=12. Verdict on Delta B is `FLAT_OR_NEGATIVE`. Both reported honestly."

**[switch to dev.to tab]**

---

### Segment 4 — Blog post (4:30 – 5:00) [30 s]

**[on screen: dev.to blog post, address bar visible]**

> "[4:30] The full story is on dev.to — `dev.to/eyorata/when-your-training-loss-is-lying-to-you-...`. The address bar shows it's live."

**[scroll smoothly through the headings — TL;DR, The gap, The dataset, The training experiment, The honest result, What's next]**

> "[4:45] About 1,800 words. Covers the gap, the audit method, the dataset, the training run including the v1-to-v2 data correction, and the honest negative Delta B."

**[switch to GitHub issue tab]**

---

### Segment 5 — Community engagement (5:00 – 5:30) [30 s]

**[on screen: github.com/sierra-research/tau2-bench/issues/295, address bar visible]**

> "[5:00] Community engagement — issue 295 on the τ²-Bench v2 repo. The URL is in the address bar. The issue presents the gap finding and links the dataset and the blog."

**[scroll to show the issue body — at minimum show the bullet list of failure modes and the example task]**

> "[5:20] Substantive content — the gap, the example failure (TB-0623 with the Go engineer over-commitment), the proposed complementary slices index. Not a placeholder."

**[switch to terminal or VS Code for the wrap]**

---

### Segment 6 — Wrap (5:30 – 6:00) [30 s]

**[on screen: VS Code with the README.md status section visible, OR the GitHub repo home page]**

> "[5:30] To recap — built a 266-task benchmark, trained a SimPO judge whose Delta A is positive but whose Delta B is negative, deployed the prompt-engineered judge in production with kill-switch triggers, and shipped four public artifacts: dataset on HuggingFace, blog on dev.to, community issue on τ²-Bench, source on GitHub."

> "[5:50] Total spend: 4 cents of a 10-dollar envelope. Repo at `github.com/eyorata/sales_evaluation_bench`. Thanks for watching."

**[stop recording]**

---

## Editing checklist (post-record, < 10 min)

- [ ] Trim silence from start and end (most recording tools do this auto with a "trim" feature).
- [ ] If you fluffed a segment by more than 3 seconds, cut and re-record JUST that segment, splice it in.
- [ ] **Do NOT add background music.** It hurts intelligibility and the rubric grades audio quality.
- [ ] Add a 1-second title card at the start (just text on solid background): "Tenacious-Bench v0.1 — Demo Walkthrough — Eyoel Nebiyu, 2026-05-02". OBS has this; in Game Bar, do it in the editor.
- [ ] Export as MP4 H.264, 1080p, 30 fps.
- [ ] Test the file plays without errors on a friend's machine.
- [ ] Upload to YouTube as **Unlisted** (not Private), or to Google Drive with "Anyone with the link can view" — verify in incognito mode that the link plays without login.
- [ ] Post the URL in your final submission and update `evidence_graph.json` `PUB-blog`-adjacent fields.

---

## If something goes wrong during recording

| Problem | Fix |
|---|---|
| Time creeping past 6:00 in segment 1 or 3 | Cut the second sentence of the narration; both segments have one optional elaboration sentence you can drop |
| HuggingFace page slow to load | Pre-load all tabs at full content; use Ctrl+Tab to swap tabs (no reload) |
| Terminal command errors during scoring demo | Run it once before recording with the exact same command — and if it fails during recording, cut to the screenshot of the JSON output instead and narrate over it |
| Mic distortion mid-recording | Better to start over than to ship a bad-audio video — re-record the offending segment only |
| Forgot to show one rubric subitem | Re-record JUST that segment and splice — don't ship a 5/6 video |
