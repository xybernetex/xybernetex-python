"""Outcome signals: what happened after a follow-up decision, reduced to
labels - the OpenClaw plugin's src/outcomes.js, same episode format.

A customer's real work has no task checker, so learning which follow-ups pay
off on it needs signals the adapter can see for itself: did the run die and
did our retry finish it; did our check-your-work turn change files ("fixed")
or only look ("confirmed"); and what the user said next - a correction, the
same request again, thanks, or something new. The user's message is
classified here, in memory, against their previous request; neither text is
ever logged or sent, only the label.

One episode per follow-up decision (including "none"). It closes on the first
of: the user's next message in that session, `quiet_s` without one, the
session ending, or a newer decision in the same session. A closed episode is
logged and, when sharing is on, sent to the policy service (POST /outcome).

Unlike the plugin, the adapter knows the follow-up's result when it opens
the episode (the follow-up runs inside `Xybernetex.run`), so there's no
start/end bookkeeping for it here.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

USER_FOLLOWUPS = ("correction", "repeat", "thanks", "new", "none", "session_end", "superseded", "evicted")
_FROM_USER = USER_FOLLOWUPS[:4]

# The plugin's write tools, plus the common SDK names (write_file, edit_file).
WRITE_TOOLS = re.compile(r"^(write|edit|multi_?edit|apply_?patch|create_?file|write_?file|edit_?file|str_replace\w*|notebook_?edit)$",
                         re.IGNORECASE)

# "That didn't work", "still failing", "you forgot the tests", "no, ...".
# Phrases about the last answer only - a new request like "fix the error in
# parse.py" or "what's wrong with my config?" is not a correction.
CORRECTION = re.compile("|".join([
    r"\b(did ?n[o']?t|does ?n[o']?t|do ?n[o']?t|is ?n[o']?t|was ?n[o']?t|are ?n[o']?t|not) (work|working|run|running|pass|passing|compile|build|fix|fixed|right|correct|done|what i (asked|wanted|meant))\b",
    r"\bstill (broken|failing|fails|wrong|missing|not|doesn|errors?|crash)",
    r"\b(that'?s|this is|it'?s|that is) (wrong|incorrect|not right|broken)",
    r"\b(you|u) (forgot|missed|skipped|ignored|broke|didn'?t)\b",
    r"\b(try again|redo it|do it again|start over|same (error|problem|issue) (again|still))\b",
    r"^\s*(nope|nah)\b",
    r"^\s*no\s*[,.!]",
]), re.IGNORECASE)
THANKS = re.compile(r"\b(thanks|thank you|thx|great|perfect|awesome|excellent|works now|that worked|it works|lgtm|looks good|good job|well done)\b",
                    re.IGNORECASE)
_LABEL_JUNK = re.compile(r"[^\w.:@/+-]+")


def as_label(value: Any) -> str | None:
    """Model ids, rules and policy names travel as labels; a rule can carry an
    error message ("decide-failed: ..."), so keep only label characters."""
    return _LABEL_JUNK.sub("_", value)[:200] if isinstance(value, str) and value else None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", str(text).lower()))


def _similarity(a: str, b: str) -> float:
    x, y = _words(a), _words(b)
    if len(x) < 3 or len(y) < 3:
        return 0.0
    shared = len(x & y)
    return shared / (len(x) + len(y) - shared)


def classify_followup(previous: str | None, text: str) -> str:
    """The user's next message, relative to their previous request."""
    text = str(text or "")
    if CORRECTION.search(text):
        return "correction"
    if previous and _similarity(previous, text) >= 0.6:
        return "repeat"
    if THANKS.search(text) and len(text.split()) <= 8:
        return "thanks"
    return "new"


def verify_result(followup: dict | None) -> str | None:
    if not followup:
        return None
    if not followup.get("success"):
        return "failed"
    return "fixed" if followup.get("writes", 0) > 0 else "confirmed"


class OutcomeTracker:
    def __init__(self, *, log: Callable[[dict], Any] = lambda e: None, send: Callable[[dict], Any] | None = None,
                 quiet_s: float = 30 * 60, max_tracked: int = 500, now: Callable[[], float] = time.monotonic,
                 set_timer: Callable[[Callable[[], None], float], Any] | None = None,
                 clear_timer: Callable[[Any], None] | None = None) -> None:
        self._log, self._send, self._quiet_s, self._max, self._now = log, send, quiet_s, max_tracked, now
        self._set_timer = set_timer or self._thread_timer
        self._clear_timer = clear_timer or (lambda t: t.cancel())
        self._episodes: dict[str, dict] = {}       # session key -> open episode
        self._last_request: dict[str, str] = {}    # session key -> the user's previous request (memory only)
        self._lock = threading.RLock()
        self._sending: list[threading.Thread] = []

    @staticmethod
    def _thread_timer(fn: Callable[[], None], seconds: float) -> threading.Timer:
        t = threading.Timer(seconds, fn)
        t.daemon = True
        t.start()
        return t

    def open(self, session_key: str, *, agent_id: str | None, model: str | None, mode: str, decision: dict,
             applied: str, run: dict, followup: dict | None) -> None:
        """After a decision on a user's run. `applied` is the action whose turn actually
        started ("none" in observe mode or when nothing was decided); `run` and
        `followup` are {success, toolCalls, tokens, ...} counters."""
        with self._lock:
            if session_key in self._episodes:
                self._close(session_key, "superseded")
            self._episodes[session_key] = {
                "agent_id": agent_id, "model": as_label(model), "mode": mode, "action": decision.get("action", "none"),
                "applied": applied, "rule": as_label(decision.get("rule")), "policy": as_label(decision.get("policy")),
                # The chance of what was actually applied. In act mode that's the
                # decision's own probability - including a "none" the rule held out,
                # which is what makes untreated runs comparable. A decision nobody
                # carried out (observe mode, a follow-up that couldn't start) left
                # the run untreated for certain.
                "probability": (decision.get("probability", 1) if mode == "act" and applied == decision.get("action", "none")
                                else 1),
                "run": run, "followup": followup if applied != "none" else None, "last_end": self._now(),
                "timer": self._set_timer(lambda: self.close(session_key, "none"), self._quiet_s),
            }
            while len(self._episodes) > self._max:
                self._close(next(iter(self._episodes)), "evicted")

    def note_user_turn(self, session_key: str, text: str) -> dict | None:
        """A user's turn (not ours): closes the open episode with how they followed
        up, then remembers this request for the next one."""
        with self._lock:
            closed = (self._close(session_key, classify_followup(self._last_request.get(session_key), text))
                      if session_key in self._episodes else None)
            self._last_request[session_key] = text
            while len(self._last_request) > self._max:
                del self._last_request[next(iter(self._last_request))]
            return closed

    def close(self, session_key: str, user: str) -> dict | None:
        with self._lock:
            return self._close(session_key, user)

    def end_session(self, session_key: str) -> dict | None:
        with self._lock:
            self._last_request.pop(session_key, None)
            return self._close(session_key, "session_end")

    def flush(self, timeout: float = 5.0) -> None:
        """Close every open episode (session_end) and wait for pending sends."""
        with self._lock:
            for key in list(self._episodes):
                self._close(key, "session_end")
            sending, self._sending = self._sending, []
        for t in sending:
            t.join(timeout)

    def open_episodes(self) -> int:
        return len(self._episodes)

    def _close(self, session_key: str, user: str) -> dict | None:
        ep = self._episodes.pop(session_key, None)
        if ep is None:
            return None
        try:
            self._clear_timer(ep["timer"])
        except Exception:  # noqa: BLE001
            pass
        followup = ep["followup"]
        episode = {
            "model": ep["model"], "mode": ep["mode"], "action": ep["action"], "applied": ep["applied"],
            "probability": ep["probability"], "rule": ep["rule"], "policy": ep["policy"],
            "run": ep["run"], "followup": followup,
            "verify": verify_result(followup) if ep["applied"] == "verify" else None,
            "user": user,
            "gapSec": round(self._now() - ep["last_end"]) if user in _FROM_USER else None,
        }
        try:
            self._log({"type": "episode", "sessionKey": session_key, "agentId": ep["agent_id"], **episode})
        except Exception:  # noqa: BLE001 - best-effort
            pass
        if self._send is not None:
            t = threading.Thread(target=self._send_quietly, args=(episode,), daemon=True)
            t.start()
            self._sending = [s for s in self._sending if s.is_alive()] + [t]
        return episode

    def _send_quietly(self, episode: dict) -> None:
        try:
            self._send(episode)
        except Exception:  # noqa: BLE001 - outcomes are best-effort telemetry
            pass
