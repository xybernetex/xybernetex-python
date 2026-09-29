"""Follow-up turns after a run ends: a retry when it died, a check-your-work
turn when it finished. The prompts are the OpenClaw plugin's (src/
interventions.js MESSAGES), so both ports send the same words, and the local
rule is the policy service's v1-uniform (2026-09-29): verify every finished
run that did work, retry every retriable death, each held out 10% of the
time so outcomes stay measurable.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Callable

MARKER = "[xybernetex]"
MESSAGES = {
    "retry": f"{MARKER} Your previous turn ended before the task was finished (the model's response couldn't be used). "
             "Continue the original task from where it stopped. Check what's already done first, then finish the rest.",
    "verify": f"{MARKER} Before I accept this, check your work against the original request:\n"
              "1. Re-read the request and confirm every part of it is done.\n"
              "2. For each file you created or changed, confirm it exists at the requested path with the expected content.\n"
              "3. If you reported a command's output or test results, re-run it and make sure your answer matches what it actually prints.\n"
              "4. Check the edge cases and exact formats the request mentions.\n"
              "If anything is missing or wrong, fix it now. Then give your final answer again in full.",
}
LOCAL_POLICY = "local-v1-uniform-2026-09-29"
P_RETRY = 0.9
P_VERIFY = 0.9


def is_ours(prompt: object) -> bool:
    """Our own follow-up turn is never the user's request."""
    return isinstance(prompt, str) and prompt.lstrip().startswith(MARKER)


@dataclass
class RunSummary:
    """The shape of an ended run - what a decision is made from. Never prompts,
    files or error text beyond a short classification."""
    success: bool
    tool_calls: int
    retriable: bool = False
    model: str | None = None
    error: str | None = None


def decide_local(summary: RunSummary, random: Callable[[], float] = _random.random,
                 p_retry: float = P_RETRY, p_verify: float = P_VERIFY) -> dict:
    """{action, probability, rule, policy}: which follow-up an ended run gets under the local rule."""
    pick = lambda action, p, rule: {"action": action, "probability": p, "rule": rule, "policy": LOCAL_POLICY}  # noqa: E731
    if not summary.success:
        if not summary.retriable:
            return pick("none", 1.0, "not-retriable")
        return pick("retry", p_retry, "retry-on-death") if random() < p_retry else pick("none", 1 - p_retry, "retry-held-out")
    if summary.tool_calls < 1:
        return pick("none", 1.0, "no-work")
    return pick("verify", p_verify, "verify") if random() < p_verify else pick("none", 1 - p_verify, "verify-held-out")
