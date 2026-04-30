"""Convert training partition to SimPO preference pairs format.

Usage:
  python -m generation_scripts.convert_to_preference_pairs
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import DATASET_DIR, write_jsonl


def load_tasks(partition: str = "train") -> list[dict]:
    """Load tasks from a partition."""
    path = DATASET_DIR / partition / "tasks.jsonl"
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def generate_chosen_response(task: dict) -> str:
    """Generate a correct (chosen) response that passes the rubric.
    
    This is a template-based generation - in production, this would
    use a dev-tier model to generate actual responses. For now, we
    generate template responses based on the expected_action.
    """
    scenario = task["input"]["scenario"]
    expected_action = task["input"].get("expected_action", "draft_outbound")
    dimension = task["dimension"]
    
    # Template responses based on dimension and action
    if expected_action == "ask_question":
        return """Subject: Re: Your inquiry

Hi,

Thanks for reaching out. I'd like to learn more about your current situation before sharing how we might help.

A few questions:
- What's your timeline for bringing on additional engineers?
- What specific technologies are you working with?
- What's driving the need right now?

Happy to schedule a call to discuss further.

Best,
[Name]
Tenacious"""
    
    elif expected_action == "escalate_human":
        return """Subject: Re: Your request

Hi,

Thank you for your interest. Your request is somewhat outside our standard offerings, and I'd like to connect you directly with our team to explore the best options for your needs.

I'll have someone reach out within 24 hours to schedule a brief call.

Best,
[Name]
Tenacious"""
    
    # Default: draft_outbound with correct behavior per dimension
    if "bench_over_commitment" in dimension:
        return """Subject: Re: Engineering team capacity

Hi,

Thank you for your interest. I wanted to address your request directly.

Based on your needs, I should mention that our current capacity may not fully align with the scale you're describing. Rather than overcommit, I'd prefer to be transparent about what we can realistically deliver:

- We have [X] engineers available in the requested stack
- We could propose a phased approach starting with [Y] engineers
- We'd need to discuss timeline flexibility to ensure quality

I'd welcome a conversation to explore what realistic options look like for your situation.

Best,
[Name]
Tenacious"""
    
    elif "icp_misclassification" in dimension:
        return """Subject: Re: Partnership inquiry

Hi,

Thank you for reaching out. I understand you're going through a significant transition, and I appreciate you sharing those details.

Given the current leadership changes and organizational priorities, I'd suggest we take a more cautious approach rather than rushing into a new partnership. Perhaps we could reconnect in 90 days when things have stabilized?

In the meantime, I'm happy to stay in touch and explore opportunities when the timing is better suited to your situation.

Best,
[Name]
Tenacious"""
    
    elif "signal_over_claiming" in dimension:
        return """Subject: Re: Your hiring needs

Hi,

Thanks for reaching out. I appreciate you sharing details about your current situation.

Rather than making assumptions about your growth trajectory, I'd prefer to learn more about your specific needs:

- How many engineers are you looking to add?
- What technologies are core to your work?
- What's driving the timeline?

This will help me understand whether we can be helpful and in what capacity.

Best,
[Name]
Tenacious"""
    
    elif "dual_control_coordination" in dimension:
        return """Subject: Re: Scheduling a call

Hi,

Thank you for your interest. Before we proceed with scheduling, I'd like to confirm a few details:

- What's the best way to reach you?
- Are there specific topics you'd like to cover?
- Who else should be involved in the conversation?

I want to ensure we're prepared to make the most of our time together.

Best,
[Name]
Tenacious"""
    
    else:
        # Generic correct response
        return """Subject: Re: Your inquiry

Hi,

Thank you for reaching out. I appreciate you sharing the context around your situation.

Based on what you've described, I'd like to learn more before proposing any specific approach. Could we schedule a brief call to discuss your needs in more detail?

Best,
[Name]
Tenacious"""


def generate_rejected_response(task: dict) -> str:
    """Generate an incorrect (rejected) response that fails the rubric.
    
    This uses the banned patterns and failure modes from the task's rubric
    to generate responses that would trigger the failure.
    """
    dimension = task["dimension"]
    scenario = task["input"]["scenario"]
    
    # Generate responses that would fail based on dimension
    if "bench_over_commitment" in dimension:
        # Would commit to more than bench has
        return """Subject: Re: Engineering team capacity

Hi,

Yes, we can absolutely deliver [10/15/20] engineers by next week. Our bench is fully available and ready to deploy immediately. We've helped companies like yours scale rapidly and we're confident we can meet your timeline.

Let's schedule a call to finalize the details.

Best,
[Name]
Tenacious"""
    
    elif "icp_misclassification" in dimension:
        # Would pitch wrong segment
        return """Subject: Congratulations on the funding + partnership opportunity

Hi,

Congratulations on the recent funding! That's exciting news. Given your rapid growth trajectory, I wanted to reach out about how we can help you scale faster than recruiting.

We've helped similar companies grow from 50 to 200 engineers in just a few months. Our approach is specifically designed for high-growth companies like yours.

Would love to schedule a call to discuss.

Best,
[Name]
Tenacious"""
    
    elif "signal_over_claiming" in dimension:
        # Would over-claim signals
        return """Subject: Re: Your aggressive hiring plans

Hi,

I can see you're scaling aggressively - everyone in your sector is doing the same thing. With your recent funding and hiring velocity, you clearly need a partner who can keep pace.

We've helped companies like yours scale rapidly. Let's talk about how we can support your growth.

Best,
[Name]
Tenacious"""
    
    elif "dual_control_coordination" in dimension:
        # Would book without confirmation
        return """Subject: Your meeting is confirmed

Hi,

I've gone ahead and booked a meeting for next Tuesday at 2pm. I've included the calendar invite with the video link.

Looking forward to discussing how we can help.

Best,
[Name]
Tenacious"""
    
    else:
        # Generic incorrect response
        return """Subject: Re: Your inquiry

Hi,

Yes, we can definitely help with that. We've worked with many companies in your sector and have the expertise to deliver results quickly.

Let's schedule a call to get started.

Best,
[Name]
Tenacious"""


def convert_to_preference_pairs(tasks: list[dict]) -> list[dict]:
    """Convert tasks to SimPO preference pairs format."""
    pairs = []
    
    for task in tasks:
        # Build prompt from input
        input_data = task["input"]
        scenario = input_data.get("scenario", "")
        hiring_brief = input_data.get("hiring_signal_brief", {})
        bench_summary = input_data.get("bench_summary", {})
        prior_thread = input_data.get("prior_thread", [])
        
        # Construct prompt
        prompt_parts = [scenario]
        
        if hiring_brief:
            prompt_parts.append(f"\nHiring Signal Brief: {json.dumps(hiring_brief)}")
        
        if bench_summary:
            prompt_parts.append(f"\nBench Summary: {json.dumps(bench_summary)}")
        
        if prior_thread:
            thread_str = "\n".join([
                f"{t.get('role', 'unknown')}: {t.get('body', '')}"
                for t in prior_thread
            ])
            prompt_parts.append(f"\nPrior Thread:\n{thread_str}")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate chosen and rejected responses
        chosen = generate_chosen_response(task)
        rejected = generate_rejected_response(task)
        
        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "task_id": task["task_id"],
            "dimension": task["dimension"],
            "difficulty": task["difficulty"],
            "source_mode": task["source_mode"],
        }
        
        pairs.append(pair)
    
    return pairs


def main():
    """Convert training partition to preference pairs."""
    print("Loading training tasks...")
    tasks = load_tasks("train")
    print(f"Loaded {len(tasks)} tasks")
    
    print("Converting to preference pairs...")
    pairs = convert_to_preference_pairs(tasks)
    print(f"Generated {len(pairs)} preference pairs")
    
    # Write to training_data directory
    output_dir = Path(__file__).resolve().parents[1] / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "preference_pairs.jsonl"
    write_jsonl(output_path, pairs)
    print(f"Wrote preference pairs to {output_path}")
    
    # Also write a JSONL with just prompts for analysis
    prompts_path = output_dir / "prompts_only.jsonl"
    prompts_only = [{"task_id": p["task_id"], "prompt": p["prompt"]} for p in pairs]
    write_jsonl(prompts_path, prompts_only)
    print(f"Wrote prompts only to {prompts_path}")
    
    return pairs


if __name__ == "__main__":
    main()