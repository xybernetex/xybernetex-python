"""The governor: deterministic stops for runs that spend without progress.

One governed run is everything a single `xyb.run` does - the first turn and
every follow-up or fix turn after it. Its limits:

  max_tool_calls      tool calls across the run
  max_tokens          model tokens across the run
  max_seconds         wall-clock time since the run started
  repeat_limit        the same call (tool + arguments) this many times in a row
  no_progress_rounds  fix rounds in a row that didn't improve the contract

When one is reached the run is stopped: the call that crossed it, and every
call after it, gets a tool result telling the model to stop and summarize
(the call itself never runs), and no follow-up or fix turn starts. Every stop
is logged with its kind and the counters - never call arguments.

Like everything outside the model, a stop depends only on what happened, so
the same run is stopped at the same point every time.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

KINDS = ("tool-calls", "tokens", "time", "repeat", "no-progress")


@dataclass(frozen=True)
class Limits:
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    repeat_limit: int | None = None
    no_progress_rounds: int | None = None


# Generous enough not to touch a healthy run on our benchmarks; tight enough to
# end the loops and the hour-long runs that end in nothing.
STANDARD = Limits(max_tool_calls=150, max_tokens=1_500_000, max_seconds=3600, repeat_limit=4, no_progress_rounds=2)


def limits_from(spec: Any) -> Limits | None:
    """None/False -> no governor; "standard" -> STANDARD; a dict or Limits -> those (a dict overrides STANDARD)."""
    if spec in (None, False):
        return None
    if spec in (True, "standard"):
        return STANDARD
    if isinstance(spec, Limits):
        return spec
    if isinstance(spec, dict):
        unknown = set(spec) - {f for f in Limits.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown governor limits: {', '.join(sorted(unknown))}")
        return replace(STANDARD, **spec)
    raise ValueError('governor must be "standard", a dict of limits, or a Limits')


def _signature(tool_name: str, params: Any) -> str:
    try:
        canon = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        canon = repr(params)
    return hashlib.sha256(f"{tool_name}\0{canon}".encode("utf-8")).hexdigest()[:16]


STOP_MESSAGES = {
    "tool-calls": "the run has used its budget of {n} tool calls",
    "tokens": "the run has used its budget of {n} model tokens",
    "time": "the run has used its time budget of {n} seconds",
    "repeat": "the same call has now been made {n} times in a row with no change",
    "no-progress": "{n} fix attempts in a row made no progress",
}


def stop_message(kind: str, n: Any) -> str:
    return (f"Stopped by Xybernetex: {STOP_MESSAGES[kind].format(n=n)}. This call was not run, and no further calls "
            "will run. Stop working now and reply with a short summary of what is done and what isn't.")


@dataclass
class _Run:
    started: float
    calls: int = 0
    tokens_done: int = 0                       # tokens of turns that have ended
    recent: list = field(default_factory=list)  # signatures of the latest calls
    stalls: int = 0                             # fix rounds in a row without improvement
    stopped: str | None = None
    message: str | None = None


class Governor:
    def __init__(self, limits: Limits, log: Callable[[dict], Any] = lambda e: None,
                 now: Callable[[], float] = time.monotonic, max_runs: int = 500) -> None:
        self.limits, self._log, self._now, self._max = limits, log, now, max_runs
        self._runs: dict[str, _Run] = {}
        self._calls: dict[str, str | None] = {}   # call id -> the verdict given to it (asked more than once)

    def start(self, session: str) -> None:
        """A new governed run in this session: counters from zero."""
        self._runs.pop(session, None)
        self._runs[session] = _Run(started=self._now())
        while len(self._runs) > self._max:
            del self._runs[next(iter(self._runs))]

    def _run(self, session: str) -> _Run:
        if session not in self._runs:
            self.start(session)
        return self._runs[session]

    def stopped(self, session: str) -> str | None:
        run = self._runs.get(session)
        return run.stopped if run else None

    def _stop(self, session: str, run: _Run, kind: str, n: Any, tokens: int | None) -> str:
        run.stopped, run.message = kind, stop_message(kind, n)
        self._log({"type": "governor_stop", "sessionKey": session, "reason": kind, "calls": run.calls,
                   "tokens": tokens, "seconds": round(self._now() - run.started, 1)})
        return run.message

    def on_call(self, session: str, call_id: str, tool_name: str, params: Any, turn_tokens: int | None = None) -> str | None:
        """Before a tool call runs: None to let it run, or the tool result to give instead.
        Asked again for the same call id, it gives the same answer."""
        if call_id and call_id in self._calls:
            return self._calls[call_id]
        verdict = self._judge(session, tool_name, params, turn_tokens)
        if call_id:
            self._calls[call_id] = verdict
            while len(self._calls) > 5000:
                del self._calls[next(iter(self._calls))]
        return verdict

    def _judge(self, session: str, tool_name: str, params: Any, turn_tokens: int | None) -> str | None:
        run, lim = self._run(session), self.limits
        if run.stopped:
            return run.message
        run.calls += 1
        tokens = run.tokens_done + (turn_tokens or 0)
        run.recent = (run.recent + [_signature(tool_name, params)])[-max(lim.repeat_limit or 1, 1):]
        if lim.max_tool_calls is not None and run.calls > lim.max_tool_calls:
            return self._stop(session, run, "tool-calls", lim.max_tool_calls, tokens)
        if lim.max_tokens is not None and tokens > lim.max_tokens:
            return self._stop(session, run, "tokens", lim.max_tokens, tokens)
        if lim.max_seconds is not None and self._now() - run.started > lim.max_seconds:
            return self._stop(session, run, "time", int(lim.max_seconds), tokens)
        if lim.repeat_limit and len(run.recent) >= lim.repeat_limit and len(set(run.recent)) == 1:
            return self._stop(session, run, "repeat", lim.repeat_limit, tokens)
        return None

    def turn_done(self, session: str, tokens: int | None) -> None:
        """A turn ended: its tokens count toward the run's budget from now on."""
        self._run(session).tokens_done += int(tokens or 0)

    def on_round(self, session: str, outcome: str) -> str | None:
        """After a fix round (the ratchet's outcome): the stop kind if the run should end, else None."""
        run, lim = self._run(session), self.limits
        run.stalls = 0 if outcome == "improved" else run.stalls + 1
        if not run.stopped and lim.no_progress_rounds and run.stalls >= lim.no_progress_rounds:
            self._stop(session, run, "no-progress", lim.no_progress_rounds, run.tokens_done)
        return run.stopped
