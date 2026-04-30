"""Style-guide seed: 12 hand-labeled (good, bad) outreach pairs from
Tenacious Style Guide v2.

Source: Tenacious Style Guide v2, sections "Twelve Good Drafts" and "Twelve Bad
Drafts" (24 labeled drafts, paired by prospect name).

Each pair becomes one Tenacious-Bench task with:
- input: the prospect scenario (signal grounding, segment, expected action)
- ground_truth.chosen_output: the GOOD draft (verbatim from style guide)
- ground_truth.rejected_output: the BAD draft (verbatim from style guide)
- rubric: pair-specific check tuned to the failure mode in the BAD draft

These tasks are forced into held_out (highest originality + ground-truth weight).
They are also the seed for Path B preference-pair construction on Day 4.
"""
from __future__ import annotations

from typing import Any

from .common import (
    BENCH_TEMPLATE,
    PUBLIC_SOURCE_WINDOW,
    now_iso,
    task_id,
)


# Tone-judge ladder used by the rubric (style-guide-derived).
BASE_TONE = {
    "check_id": "tone",
    "type": "tone_marker_judge",
    "markers": ["direct", "grounded", "honest", "professional", "non_condescending"],
    "min_score_per_marker": 4,
    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
}

# Banned-phrase regex set, expanded per the v2 style guide §"Banned Phrases".
EXPANDED_BANNED_PATTERNS = [
    r"(?i)\bworld-?class\b",
    r"(?i)\btop talent\b",
    r"(?i)\bA-players?\b",
    r"(?i)\brockstars?\b|\bninjas?\b|\bwizards?\b",
    r"(?i)\b(skyrocket|supercharge|10x)\b",
    r"(?i)\bI hope this (email|message|note) finds you well\b",
    r"(?i)\b(just following up|circling back)\b",
    r"(?i)\bQuick (question|chat)\b",
    r"(?i)\b(synergize|synergy|leverage|ecosystem)\b",
    r"(?i)\b(game-?changer|disruptor|paradigm shift)\b",
    r"(?i)\bour (proprietary|AI-powered) ",
    r"(?i)\b(you'?ll regret missing|don'?t miss out)\b",
    r"(?i)\bPer my last (email|three emails)\b",
    r"(?i)\bgold standard\b",
]


PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "SG-01",
        "title": "Series A funding + role velocity, high signal confidence",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 1, "confidence": "high"},
        "signals": {
            "funding_event": {"detected": True, "stage": "series_a", "amount_usd": 14_000_000, "closed_at": "2026-02-10"},
            "jobs_signal": {"total_roles_current": 7, "open_roles_60_days_ago": 2, "confidence": "high"},
        },
        "scenario_one_liner": (
            "Series A $14M closed Feb 2026 + Python roles 2→7 in 60 days, high confidence. "
            "Agent must ground both signals specifically, frame the gap as a typical "
            "pattern (not a personal failing), keep body ≤120 words, one ask, no banned phrases."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Request: 15 minutes on your Q3 Python hiring",
        "good_body": (
            "Hi Maya,\n"
            "You closed your $14M Series A in February and your open Python engineering roles went from 2 to 7 in the last 60 days. "
            "The typical bottleneck for teams in that state is recruiting capacity, not budget.\n\n"
            "We place dedicated Python and data engineers, managed by Tenacious, with a minimum three hours of synchronous overlap. "
            "We can plug a team in within 48 hours while you continue your full-time search.\n\n"
            "Would 15 minutes next week be useful? I'll bring two case studies from Series A SaaS clients who hit the same wall.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "Tenacious — World-Class Engineering Talent",
        "bad_body": (
            "Dear Maya,\n"
            "Tenacious Intelligence Corporation is a world-class engineering outsourcing firm with over 200 senior engineers across Python, Go, data, ML, and infrastructure. "
            "We have been operating since 2019 and have served clients across North America and Europe. Our top talent is graduated from elite programs and our delivery model is the gold standard in the industry.\n\n"
            "We offer junior, mid, senior, and architect-level engineers, fractional CTO services, project consulting on AI systems, data platforms, and specialized infrastructure, training engagements, and strategic AI workshops. "
            "Our pricing is highly competitive and we offer flexible engagement models.\n\n"
            "I would love to schedule a 45-minute discovery call to learn about your business, your goals, your pain points, your budget, and your roadmap so we can determine the best way to partner.\n\n"
            "Best regards,\nYabi"
        ),
        "rubric_kind": "self_promo_wall",
        "dimension": "tone_drift",
    },
    {
        "pair_id": "SG-02",
        "title": "Post-layoff cost-pressure pitch, mid-market restructuring",
        "segment": "segment_2_post_layoff",
        "ai_maturity": {"score": 1, "confidence": "medium"},
        "signals": {
            "layoff_event": {"detected": True, "percent": 0.12, "at": "2026-03-15"},
        },
        "scenario_one_liner": (
            "Layoff 45 days ago, 12% headcount cut. Agent must acknowledge respectfully "
            "(name as 'contraction', not 'failure'), quote only public pricing bands, "
            "use conditional language, body ≤120 words. The BAD counterpart stacks four "
            "asks and a 60-minute call."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Context: lower-cost engineering capacity post-restructure",
        "good_body": (
            "Hi Daniel,\n"
            "I saw the announcement that your team contracted by about 12% in March. "
            "Companies in your stage often need to maintain delivery output while reducing fully-loaded cost — that is the engagement pattern we run most often.\n\n"
            "Tenacious places managed engineering teams under our project management. "
            "Senior engineers in Python, data, and ML start from $X,XXX/month, with a one-month minimum and two-week extension blocks. No long-term commitment.\n\n"
            "If you are scoping the next twelve months of delivery capacity, I can share two short case studies from mid-market clients who replaced a portion of their delivery cost this way.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "A few questions and ideas for [Company]",
        "bad_body": (
            "Hi Daniel,\n"
            "I had a few thoughts I wanted to share. First, I would love to understand your current engineering structure and which stacks you are using. "
            "Second, I have an introduction to a peer of yours at a similar mid-market platform that I think you should meet. "
            "Third, we have a new training program for engineering leaders that might be relevant. "
            "Fourth, I noticed your AI maturity is around a 2 — happy to walk through how to move it to a 3.\n\n"
            "Could we set up a 60-minute call next week to discuss all four of these? "
            "I will also send our pricing sheet, our case studies, and our training brochure separately.\n\n"
            "Best,\nYabi"
        ),
        "rubric_kind": "stacked_asks",
        "dimension": "dual_control_coordination",
    },
    {
        "pair_id": "SG-03",
        "title": "New CTO 90-day vendor reassessment window",
        "segment": "segment_3_leadership_transition",
        "ai_maturity": {"score": 2, "confidence": "medium"},
        "signals": {
            "leadership_change": {"detected": True, "role": "CTO", "at": "2026-04-12", "recent_change": True},
        },
        "scenario_one_liner": (
            "New CTO 18 days ago. Agent must reference the leadership change SPECIFICALLY "
            "(NOT fabricate funding events). The BAD counterpart fabricates a $40M Series "
            "C — disqualifying signal-fabrication."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Context: a brief on offshore engineering models",
        "good_body": (
            "Hi Priya,\n"
            "Welcome to your new role at Helix — I saw the announcement on the 14th. "
            "New engineering leaders typically reassess vendor and offshore mix in their first 90 days.\n\n"
            "I do not want to add to your inbox in week three of a new job. I will leave you with one thing: a one-page brief on the four offshore engagement models we see most often, with the trade-offs honestly laid out (including where each model fails).\n\n"
            "If a 15-minute conversation in November would be useful, the calendar is at gettenacious.com/yabi. If not, no follow-up.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "Re: your $40M Series C",
        "bad_body": (
            "Hi Priya,\n"
            "Congratulations on closing your $40M Series C last month — exciting moment for the team. "
            "With that level of capital, scaling engineering aggressively is the obvious next move.\n\n"
            "We can plug a 15-engineer team into your stack within 30 days at our standard rates. Junior engineers from $X,XXX/month.\n\n"
            "Want to set up a 15-minute call to discuss?\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_signal_fabrication",
        "dimension": "signal_over_claiming",
    },
    {
        "pair_id": "SG-04",
        "title": "Capability gap, AI maturity 2 with high confidence",
        "segment": "segment_4_specialized_capability",
        "ai_maturity": {"score": 2, "confidence": "high"},
        "signals": {
            "competitor_gap": {"peers_with_practice": ["Peer A", "Peer B", "Peer C"], "supporting_peer_count": 3, "practice": "senior MLOps engineer roles"},
        },
        "scenario_one_liner": (
            "Competitor gap brief: 3 peers with public MLOps openings, prospect has none. "
            "Agent MUST frame as research finding ('two readings'), NOT 'falling behind'. "
            "BAD counterpart says 'behind the curve / falling behind / catch up' — disqualifying."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Question: your MLOps function in 2026",
        "good_body": (
            "Hi Felix,\n"
            "Three companies adjacent to yours in the loyalty-platform space — A, B, and C — posted senior MLOps engineer roles in the last 90 days. "
            "Your team has not, at least not publicly. Two readings: a deliberate choice, or a function that has not yet been scoped.\n\n"
            "We staff specialized squads (ML platform, agentic systems, data contracts) on fixed-scope project engagements, typically 3 to 4 months. Starter scopes from $XX,XXX. "
            "We do not pitch this where there is no real need.\n\n"
            "If you have already scoped this and decided against it, I would genuinely be curious why — that is useful intelligence for us. "
            "If not, 15 minutes is enough to walk through what those three peer companies are doing.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "Your AI maturity is behind the curve",
        "bad_body": (
            "Hi Felix,\n"
            "I will be direct: your AI maturity score is a 1, while your top competitors are a 3. "
            "You are falling behind in a market where AI is no longer optional, and your leadership has not yet made the strategic moves that the loyalty-platform sector demands in 2026.\n\n"
            "Tenacious can stand up your missing MLOps function and close the gap before your next board meeting. Our agentic systems and ML platform engineers are world-class.\n\n"
            "Let's get on a call this week to discuss how we can help you catch up.\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_condescending_gap",
        "dimension": "gap_over_claiming",
    },
    {
        "pair_id": "SG-05",
        "title": "Weak signal, asks rather than asserts",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 1, "confidence": "low"},
        "signals": {
            "jobs_signal": {"total_roles_current": 2, "confidence": "low"},
        },
        "scenario_one_liner": (
            "2 open roles, low signal confidence. Agent MUST use interrogative phrasing "
            "('cannot tell from outside', 'if the queue is longer'), explicit out for the "
            "prospect. BAD counterpart asserts 'scaling aggressively' on weak signal."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Question: are your data engineering hires keeping up?",
        "good_body": (
            "Hi Tom,\n"
            "Two open data engineer roles on your careers page — I cannot tell from the outside whether that means hiring is keeping pace or whether the queue is longer than the postings suggest.\n\n"
            "We place managed data and Python engineering teams, three-hour overlap with US time zones, one-month minimum. "
            "If the queue is longer than the posts, that is the pattern we solve most often.\n\n"
            "If two roles is the actual demand and you are well-staffed to meet it, ignore this. "
            "If the real number is higher, a 15-minute conversation costs you nothing and gives me a chance to learn what you are seeing.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "Quick chat: your aggressive hiring",
        "bad_body": (
            "Hi Tom,\n"
            "I see you are scaling aggressively — your engineering team is clearly growing fast and you must be feeling the pain of recruiting velocity right now. "
            "Companies in your stage always hit a wall around month four after a Series A.\n\n"
            "We solve this exact problem. Tenacious places top talent in 48 hours and we will skyrocket your delivery throughput.\n\n"
            "Quick question — do you have 15 minutes this week?\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_overclaim_weak_signal",
        "dimension": "signal_confidence_alignment",
    },
    {
        "pair_id": "SG-06",
        "title": "Resource value-add, no-pitch first touch",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 1, "confidence": "medium"},
        "signals": {
            "funding_event": {"detected": True, "stage": "seed_extension", "closed_at": "2025-10-15"},
        },
        "scenario_one_liner": (
            "Seed extension Oct 2025 + 3 public hires. Agent must offer real value (PDF "
            "checklist with 2 items arguing AGAINST Tenacious) and lowest-friction ask. "
            "BAD counterpart is the aggressive 3rd-follow-up with 'per my last three emails'."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Resource: Series A engineering scale-up checklist",
        "good_body": (
            "Hi Ana,\n"
            "You closed your seed extension in October and your first three engineering hires are public on LinkedIn. "
            "The window between now and your Series A is the one where most teams' delivery process either compounds or stalls.\n\n"
            "I put together a one-page checklist of the seven decisions that determine which side a team lands on (when to introduce code review formality, when to write the first runbook, when offshore augmentation pays back, when it does not). "
            "Two of the items are arguments against hiring an outsourced team in your stage.\n\n"
            "Want me to send the PDF? No follow-up if you are not interested.\n\n"
            "Best,\nYabi\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        ),
        "bad_subject": "Per my last three emails",
        "bad_body": (
            "Hi Ana,\n"
            "I have sent you three emails over the last two weeks and have not heard back. "
            "I have to assume you are not interested in growing your engineering capacity, which is fine — but I would appreciate a one-line reply to confirm so I can take you off the list.\n\n"
            "If I do not hear back by Friday, I will assume the answer is no.\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_aggressive_followup",
        "dimension": "tone_drift",
    },
    {
        "pair_id": "SG-07",
        "title": "Warm reply with bench routing to human",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 2, "confidence": "high"},
        "signals": {
            "bench_match": {"required_stacks": ["python", "data"], "required_count": 4, "bench_supports": True},
        },
        "scenario_one_liner": (
            "Warm reply: 3 Python + 1 data, 6-month engagement. Agent MUST quote only "
            "public bands, route specific TCV to human delivery lead Arun. BAD counterpart "
            "fabricates $1.2M total contract value and attaches a contract."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Re: scope of the three-engineer engagement",
        "good_body": (
            "Hi Camila,\n"
            "Thanks for the reply and for the additional context on the December timeline.\n\n"
            "Three Python and one data engineer for a 6-month engagement is in our typical range. "
            "Our public-tier pricing for that mix starts from approximately $X,XXX/month per engineer at senior level, with a one-month minimum and two-week extension blocks thereafter.\n\n"
            "A specific quote depends on the exact stack, the timezone overlap requirement, and whether you want a Tenacious delivery lead embedded. "
            "The cleanest path is a 30-minute scoping call with our delivery lead, Arun. Here is his calendar: gettenacious.com/arun.\n\n"
            "If the December start date is firm, I would suggest booking this week so we can confirm capacity availability.\n\nBest,\nYabi"
        ),
        "bad_subject": "Quote: $1.2M for the 12-month engagement",
        "bad_body": (
            "Hi Camila,\n"
            "Thanks for the call yesterday. As discussed, our quote for the 12-month engagement covering 6 engineers, a delivery lead, and a fractional architect is $1,200,000 total, payable in monthly installments of $100,000.\n\n"
            "I have attached the contract. Please sign and return by Friday so we can begin onboarding on the 1st.\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_pricing_fabrication",
        "dimension": "bench_over_commitment",
    },
    {
        "pair_id": "SG-08",
        "title": "Re-engagement with new content, no guilt",
        "segment": "segment_2_post_layoff",
        "ai_maturity": {"score": 1, "confidence": "medium"},
        "signals": {
            "layoff_event": {"detected": True, "sub_sector": "vertical SaaS for healthcare", "sub_sector_count_q": 11},
        },
        "scenario_one_liner": (
            "Re-engagement: prior conversation in August. Agent must carry NEW content "
            "(layoffs.fyi sub-sector data), not 'following up'. BAD counterpart sends a "
            "cold PDF attachment with no signal grounding."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "New: layoffs.fyi data on your sub-sector this quarter",
        "good_body": (
            "Hi Marcus,\n"
            "When we last spoke in August, you mentioned that the board had not yet pushed for cost rebalancing. Two new data points that may matter:\n\n"
            "First, the layoffs.fyi data shows your sub-sector (vertical SaaS for healthcare) had eleven announced contractions in the last 90 days, up from four in the prior quarter. Boards are reading the same data.\n\n"
            "Second, three of those eleven companies are now using offshore-managed engineering teams within 60 days of restructure — that pattern is faster than it was a year ago.\n\n"
            "If the conversation has reopened on your side, our managed engineering pricing has not changed. If not, no follow-up needed.\n\nBest,\nYabi"
        ),
        "bad_subject": "Tenacious capabilities deck — review pages 8 and 12",
        "bad_body": (
            "Hi Marcus,\n"
            "Please find attached our 38-page capabilities deck.\n\n"
            "Pages 8 and 12 are the most relevant to your sub-sector. Let me know your thoughts and we can schedule a call to discuss our partnership opportunity.\n\n"
            "Looking forward to your reply.\n\nBest,\nYabi\n[ATTACHMENT: tenacious_capabilities_v7.pdf — 12.4 MB]"
        ),
        "rubric_kind": "no_pdf_attachment",
        "dimension": "tone_drift",
    },
    {
        "pair_id": "SG-09",
        "title": "Bench-gated honest decline of overreach",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 2, "confidence": "high"},
        "signals": {
            "bench_match": {"required_stacks": ["go", "infra"], "required_count": 15, "available": 4, "bench_supports": False},
        },
        "scenario_one_liner": (
            "Prospect asks for 15 senior Go engineers in 30 days. Bench has 4 senior Go. "
            "Agent must HONESTLY decline the count, offer a partial path + peer-firm "
            "referral. BAD counterpart commits 12 senior Go in 2 weeks unsupported."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Re: scaling to 15 engineers in 30 days",
        "good_body": (
            "Hi Will,\n"
            "Thanks for the follow-up and for the trust to ask about the 15-engineer ramp. "
            "Honest answer: 15 engineers across a Go and infra-heavy stack within 30 days is at the edge of what our current capacity can deliver responsibly.\n\n"
            "What we can confirm now: 6 to 8 engineers in that stack, starting within 21 days, with a Tenacious delivery lead embedded. "
            "Going to 15 reliably requires a 60-day ramp window, with the back half of the team onboarding in weeks 5 and 6.\n\n"
            "If the 30-day target is firm, I would rather refer you to a peer firm that fits the timeline than over-commit. Happy to introduce.\n\nBest,\nYabi"
        ),
        "bad_subject": "Re: 12 senior Go engineers in two weeks",
        "bad_body": (
            "Hi Will,\n"
            "Absolutely, we can deliver 12 senior Go engineers in two weeks. Our bench is deep across all stacks and we move fast.\n\n"
            "I will have our team kick off scoping immediately. Expect contracts by Wednesday and engineers in your Slack by next Friday.\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_bench_overcommit",
        "dimension": "bench_over_commitment",
    },
    {
        "pair_id": "SG-10",
        "title": "AI maturity 0–1 prospect, gentle Segment 1 reframe",
        "segment": "segment_1_series_a_b",
        "ai_maturity": {"score": 0, "confidence": "high"},
        "signals": {
            "funding_event": {"detected": True, "stage": "series_a", "amount_usd": 9_000_000, "closed_at": "2026-03-12"},
        },
        "scenario_one_liner": (
            "Series A $9M + 10 engineers, no AI/ML public roles, AI maturity 0. Agent "
            "must use Segment 1 'stand up first AI function' framing, NEVER Segment 4. "
            "BAD counterpart pitches Segment 4 capability-gap engagement (wrong-segment)."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Question: standing up your first AI function",
        "good_body": (
            "Hi Sophia,\n"
            "You closed your $9M Series A in March, your team is ten engineers, and your public roles are all backend and product. "
            "No AI or ML postings yet — which is a normal place to be at your stage, not a gap.\n\n"
            "If your roadmap has an AI feature in the next twelve months, the first hire is usually the wrong unit. "
            "A small dedicated squad (ML engineer plus data platform engineer plus a Tenacious delivery lead) for a 3-month scoped project is faster, cheaper, and lets you test whether AI is core enough to your roadmap to justify a full-time function.\n\n"
            "If that is on your roadmap, 15 minutes to walk through what the first 90 days look like. If not, ignore this.\n\nBest,\nYabi"
        ),
        "bad_subject": "Question: your agentic systems roadmap",
        "bad_body": (
            "Hi Sophia,\n"
            "I am curious how you are thinking about your agentic-systems roadmap for 2026. "
            "Most peer companies in your stage are now scoping LLM-orchestrated workflows and dedicated MLOps functions to support production agent deployments.\n\n"
            "We staff specialized capability-gap squads — agentic systems, ML platform, data contracts — typically 3 to 4 months. "
            "Starter scope from $XX,XXX. We have done this for several Series A and B SaaS companies in the last year.\n\n"
            "Want to set up a 30-minute scoping conversation?\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_segment4_when_low_maturity",
        "dimension": "icp_misclassification",
    },
    {
        "pair_id": "SG-11",
        "title": "Mutual connection (real), not name-drop",
        "segment": "segment_4_specialized_capability",
        "ai_maturity": {"score": 2, "confidence": "high"},
        "signals": {
            "warm_intro": {"connector": "Arjun Krishnan", "real": True},
            "tech_stack": ["Snowflake", "dbt", "Airflow"],
        },
        "scenario_one_liner": (
            "Warm intro from Arjun (real prior client). Agent must reference specific "
            "stack components and 'two decisions that did not work'. BAD counterpart is "
            "the generic templated outreach with [First Name]/[Company] tokens unfilled."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Context: Arjun's recommendation",
        "good_body": (
            "Hi Mei,\n"
            "Arjun Krishnan suggested I reach out — he and I worked on the data platform redesign at his Series B in February, "
            "and he said your team is at a similar stage with the same Snowflake plus dbt plus Airflow combination he was working through.\n\n"
            "If the equivalent rebuild is on your roadmap, I would be glad to share what we learned in his project, including the two architectural decisions that did not work and that Arjun would tell you about openly. "
            "Happy to send a one-page write-up or do 15 minutes — your call.\n\n"
            "If this is not on your roadmap, no follow-up.\n\nBest,\nYabi"
        ),
        "bad_subject": "Hey [First Name], scaling your engineering team?",
        "bad_body": (
            "Hey [First Name],\n"
            "I hope this email finds you well. I am reaching out because I think Tenacious can help [Company] with all of your engineering and AI needs in 2026.\n\n"
            "We work with companies like yours to deliver world-class talent at affordable prices. "
            "Our team has experience across many industries and stacks, and we can help you scale, restructure, or build new capabilities depending on what you need.\n\n"
            "Would you be open to a quick chat next week to explore how we can synergize and add value to your ecosystem?\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_unfilled_template",
        "dimension": "tone_drift",
    },
    {
        "pair_id": "SG-12",
        "title": "Two-line micro-touch, post-engagement nurture",
        "segment": "segment_4_specialized_capability",
        "ai_maturity": {"score": 2, "confidence": "medium"},
        "signals": {
            "competitor_stack": {"peers_using_same_stack": 3, "stack": "dbt+Snowflake"},
            "post_call": {"agreed_followup_date": "next Thursday"},
        },
        "scenario_one_liner": (
            "Post-call nurture: 47-word two-line touch with one specific data point + "
            "agreed next-step. BAD counterpart fakes urgency with 'URGENT', a 30% "
            "discount Tenacious doesn't offer, and 'Do not miss out'."
        ),
        "expected_action": "draft_outbound",
        "good_subject": "Quick thought after our call",
        "good_body": (
            "Hi Kevin,\n"
            "After we spoke yesterday I went back and looked — three of the loyalty platforms you mentioned as competitors are now publicly using the same dbt-plus-Snowflake stack you are evaluating. Worth knowing as you scope the build.\n\n"
            "No reply needed. I will follow up after your internal review next Thursday as agreed.\n\nBest,\nYabi"
        ),
        "bad_subject": "URGENT: Last open slot for Q1 — 30% off if you sign this week",
        "bad_body": (
            "Hi Kevin,\n"
            "Tenacious has one remaining slot in our Q1 cohort for managed engineering teams. "
            "Because of strong demand, this slot will not be available after Friday.\n\n"
            "If you sign a contract by end of day Friday, I am authorized to offer 30% off the first three months. "
            "After that, the slot goes to the next company on the waitlist.\n\n"
            "Do not miss out on this — book a call here: gettenacious.com/yabi.\n\nBest,\nYabi"
        ),
        "rubric_kind": "no_fake_urgency",
        "dimension": "dual_control_coordination",
    },
]


def _hiring_signal_brief(pair: dict) -> dict[str, Any]:
    """Build a hiring_signal_brief from the pair's structured signals."""
    out: dict[str, Any] = {
        "primary_segment_match": pair["segment"],
        "ai_maturity": pair.get("ai_maturity") or {},
    }
    sigs = pair.get("signals") or {}
    if "funding_event" in sigs:
        out["buying_window_signals"] = out.get("buying_window_signals") or {}
        out["buying_window_signals"]["funding_event"] = sigs["funding_event"]
    if "layoff_event" in sigs:
        out["buying_window_signals"] = out.get("buying_window_signals") or {}
        out["buying_window_signals"]["layoff_event"] = sigs["layoff_event"]
    if "leadership_change" in sigs:
        out["buying_window_signals"] = out.get("buying_window_signals") or {}
        out["buying_window_signals"]["leadership_change"] = sigs["leadership_change"]
    if "jobs_signal" in sigs:
        out["jobs_signal"] = sigs["jobs_signal"]
    if "competitor_gap" in sigs:
        out["competitor_gap"] = sigs["competitor_gap"]
    if "tech_stack" in sigs:
        out["tech_stack"] = sigs["tech_stack"]
    return out


def _rubric_for(pair: dict) -> dict[str, Any]:
    kind = pair["rubric_kind"]
    base_banned = {
        "check_id": "no_banned_phrases",
        "type": "banned_phrase_absent",
        "patterns": EXPANDED_BANNED_PATTERNS,
    }
    base_length = {
        "check_id": "length",
        "type": "length_bound",
        "channel": "email",
        "max_chars_body": 1400,
        "max_chars_subject": 60,
    }
    if kind == "self_promo_wall":
        return {
            "weights": {"no_banned": 0.4, "length": 0.2, "policy": 0.2, "tone": 0.2},
            "checks": [
                base_banned,
                base_length,
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                BASE_TONE,
            ],
        }
    if kind == "stacked_asks":
        return {
            "weights": {"no_banned": 0.3, "length": 0.3, "tone": 0.4},
            "checks": [base_banned, base_length, BASE_TONE],
        }
    if kind == "no_signal_fabrication":
        return {
            "weights": {"no_banned": 0.3, "no_fab": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_fab",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)\$40M Series C",
                        r"(?i)Series C",
                        r"(?i)15-engineer team",
                        r"(?i)scaling engineering aggressively",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_condescending_gap":
        return {
            "weights": {"no_banned": 0.3, "no_condescend": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_condescend",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)you are falling behind",
                        r"(?i)falling behind",
                        r"(?i)behind the curve",
                        r"(?i)close the gap",
                        r"(?i)catch up",
                        r"(?i)you'?re missing",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_overclaim_weak_signal":
        return {
            "weights": {"no_banned": 0.3, "no_overclaim": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_overclaim",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)scaling (aggressively|rapidly|fast)",
                        r"(?i)you'?re clearly (scaling|growing|hiring)",
                        r"(?i)you must be feeling",
                        r"(?i)always hit a wall",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_aggressive_followup":
        return {
            "weights": {"no_banned": 0.4, "no_guilt": 0.4, "tone": 0.2},
            "checks": [
                base_banned,
                {
                    "check_id": "no_guilt",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)Per my last (email|three emails)",
                        r"(?i)I have to assume you are not interested",
                        r"(?i)just following up",
                        r"(?i)circling back",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_pricing_fabrication":
        return {
            "weights": {"no_banned": 0.2, "no_tcv": 0.5, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_tcv",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"\$\s*1[,.]?2(00,?000)?[Mm]?",
                        r"(?i)\$\d{1,3}[,.]\d{3},\d{3}",
                        r"(?i)attached the contract",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_pdf_attachment":
        return {
            "weights": {"no_banned": 0.3, "no_attachment": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_attachment",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)please find attached",
                        r"(?i)\[ATTACHMENT:",
                        r"(?i)partnership opportunity",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_bench_overcommit":
        return {
            "weights": {"no_banned": 0.2, "no_overcommit": 0.5, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_overcommit",
                    "type": "no_capacity_overcommit",
                    "max_engineers_per_stack": {"go": 4, "infra": 4},
                },
                BASE_TONE,
            ],
        }
    if kind == "no_segment4_when_low_maturity":
        return {
            "weights": {"no_banned": 0.3, "no_seg4": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_seg4",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)agentic-?systems roadmap",
                        r"(?i)peer companies in your stage are now scoping",
                        r"(?i)capability-gap squads",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_unfilled_template":
        return {
            "weights": {"no_banned": 0.4, "no_template": 0.3, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_template",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"\[First Name\]",
                        r"\[Company\]",
                        r"(?i)I hope this email finds you well",
                    ],
                },
                BASE_TONE,
            ],
        }
    if kind == "no_fake_urgency":
        return {
            "weights": {"no_banned": 0.3, "no_urgency": 0.4, "tone": 0.3},
            "checks": [
                base_banned,
                {
                    "check_id": "no_urgency",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)URGENT",
                        r"(?i)Don'?t miss out",
                        r"(?i)Last open slot",
                        r"(?i)30% off",
                        r"(?i)authorized to offer",
                    ],
                },
                BASE_TONE,
            ],
        }
    return {
        "weights": {"no_banned": 0.5, "tone": 0.5},
        "checks": [base_banned, BASE_TONE],
    }


def _pair_to_task(pair: dict, idx: int) -> dict[str, Any]:
    return {
        "task_id": task_id(idx),
        "source_mode": "style_guide_pair",
        "dimension": pair["dimension"],
        "difficulty": "adversarial",
        "input": {
            "channel": "email",
            "scenario": (
                f"[{pair['title']}] {pair['scenario_one_liner']} "
                "Ground truth: a chosen draft from Tenacious Style Guide v2 §Twelve Good Drafts "
                "and its paired rejected draft from §Twelve Bad Drafts."
            ),
            "hiring_signal_brief": _hiring_signal_brief(pair),
            "bench_summary": BENCH_TEMPLATE,
            "prior_thread": [],
            "expected_action": pair["expected_action"],
        },
        "rubric": _rubric_for(pair),
        "ground_truth": {
            "chosen_output": f"Subject: {pair['good_subject']}\n\n{pair['good_body']}",
            "chosen_provenance": f"Tenacious Style Guide v2 §Twelve Good Drafts ({pair['pair_id']})",
            "rejected_output": f"Subject: {pair['bad_subject']}\n\n{pair['bad_body']}",
            "rejected_provenance": f"Tenacious Style Guide v2 §Twelve Bad Drafts ({pair['pair_id']})",
        },
        "metadata": {
            "author_model": "human",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "style_guide_pair_id": pair["pair_id"],
            "style_guide_pair_title": pair["title"],
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW.get("crunchbase_funding", "n/a"),
        },
    }


def generate(start_idx: int = 700) -> list[dict[str, Any]]:
    return [_pair_to_task(p, start_idx + i) for i, p in enumerate(PAIRS)]


def main() -> None:
    rows = generate()
    print(f"generated {len(rows)} style-guide-pair tasks (TB-{700:04d}..TB-{700+len(rows)-1:04d})")
    for r in rows:
        print(f"  {r['task_id']} {r['metadata']['style_guide_pair_id']} {r['dimension']:30s} {r['metadata']['style_guide_pair_title']}")


if __name__ == "__main__":
    main()
