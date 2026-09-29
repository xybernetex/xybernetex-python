"""Runs that died even though the framework reported success, from the run's
transcript. A port of the death detection in the OpenClaw plugin's
src/interventions.js (0.4.1).

Frameworks report a run as successful whenever the model call itself didn't
fail. The commonest deaths don't look like failures:

- the model returned nothing usable (an empty final message, stopReason
  "length" in every case seen);
- the model's final answer was cut off at its output limit and the runtime
  substituted a reply of its own ("The tool run finished, but no final
  summary was produced...", marked in OpenClaw by an idempotencyKey ending in
  ":settled-finalization-fallback"). 5 of 5 graded runs ending this way had
  failed their task; it was half of the failures on outside benchmark tasks.

Messages are plain dicts in the OpenClaw transcript shape: role, content (a
string or a list of {type, text} / {type: "toolCall"} parts), stopReason,
errorMessage, idempotencyKey. Adapters translate their framework's items to
this shape before calling in.
"""
from __future__ import annotations

import re
from typing import Any

# Failures worth a retry: the model gave nothing usable. Not user aborts, not
# approval denials, not policy blocks.
RETRIABLE = re.compile(r"incomplete_turn|format|timed?\s*out|timeout|overloaded|rate.?limit|stream|empty (response|output)"
                       r"|unusable|no final answer", re.I)
NOT_RETRIABLE = re.compile(r"abort|cancel|denied|approval|blocked|policy", re.I)

FALLBACK_KEY = re.compile(r":settled-finalization-fallback$")
FALLBACK_TEXT = re.compile(r"^The tool run finished, but no final (summary|answer) was produced", re.I)


def retriable(error: Any) -> bool:
    e = str(error if error is not None else "")
    return RETRIABLE.search(e) is not None and NOT_RETRIABLE.search(e) is None


def reply_text(m: Any) -> str:
    content = m.get("content") if isinstance(m, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(c.get("text") or "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def is_fallback_reply(m: Any) -> bool:
    if not isinstance(m, dict) or m.get("role") != "assistant":
        return False
    key = m.get("idempotencyKey")
    return (isinstance(key, str) and FALLBACK_KEY.search(key) is not None) or FALLBACK_TEXT.match(reply_text(m).strip()) is not None


def _said(m: dict) -> bool:
    content = m.get("content")
    if isinstance(content, str):
        return content.strip() != ""
    return isinstance(content, list) and any(
        isinstance(c, dict) and (c.get("type") == "toolCall" or (c.get("type") == "text" and str(c.get("text") or "").strip()))
        for c in content)


def empty_run_error(messages: Any) -> str | None:
    """The error to use in place of a reported success when the run's last
    reply is the runtime's fallback, or nothing the model said since the last
    user message has any text or tool call. None when the run did answer, or
    when there's no transcript to judge from."""
    if not isinstance(messages, list) or not messages:
        return None
    start = len(messages)
    while start > 0 and (not isinstance(messages[start - 1], dict) or messages[start - 1].get("role") != "user"):
        start -= 1
    if start == 0 and (not isinstance(messages[0], dict) or messages[0].get("role") != "user"):
        return None  # no user turn in view: can't tell
    replies = [m for m in messages[start:] if isinstance(m, dict) and m.get("role") == "assistant"]
    if not replies:
        return None
    if is_fallback_reply(replies[-1]):
        real = next((m for m in reversed(replies) if not is_fallback_reply(m)), None)
        stop = (real.get("stopReason") if real else None) or "unknown"
        return f"no final answer: OpenClaw substituted its fallback reply (stopReason {stop})"
    if any(_said(m) for m in replies):
        return None
    return f"empty response from the model (stopReason {replies[-1].get('stopReason') or 'unknown'})"


def last_reply_error(messages: Any) -> str | None:
    """The other half: a run the runtime ends as aborted comes with success
    false and no error at all, but its final assistant message says why - e.g.
    errorMessage "request timed out" (worth a retry) versus a user's stop
    ("aborted": never retried, see NOT_RETRIABLE)."""
    if not isinstance(messages, list):
        return None
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            return None
        if m.get("role") == "assistant":
            err = m.get("errorMessage")
            return err[:200] if isinstance(err, str) and err else None
    return None
