"""What every framework adapter shares: configuration, the gate and its
per-call decisions, the log, the user's own words, recording what a tool call
did, the follow-up decision and the outcome episode. An adapter supplies only
the framework's hooks (where a tool call can be judged, how a run is started
and read) - see openai_agents/ and langgraph/.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core.authz import AuthorizationTracker
from .core.control import ToolGate
from .core.followups import RunSummary, decide_local, is_ours
from .core.outcomes import WRITE_TOOLS, OutcomeTracker
from .core.policy import DEFAULT_ENDPOINT, OutcomeSender, RemoteDecider

DEFAULT_LOG = Path.home() / ".xybernetex" / "events.jsonl"


@dataclass
class Report:
    """What `run` returns: the framework's result plus what Xybernetex did with it."""
    result: Any = None                      # the framework's result, or None when the run raised
    status: str = "done"                    # done | died | held | failed
    error: str | None = None
    summary: RunSummary | None = None
    decision: dict | None = None            # the follow-up decision, if follow-ups are configured
    followup: Any = None                    # the follow-up turn's result, if one ran
    seconds: float = 0.0
    interruptions: list = field(default_factory=list)


class AdapterBase:
    def __init__(self, *, mode: str = "observe", preset: str | None = "recommended", rules: list[dict] | None = None,
                 log: Callable[[dict], Any] | str | os.PathLike | None = None, approvals: bool = True,
                 followups: dict | None = None, agent_id: str | None = None, api_key: str | None = None,
                 endpoint: str | None = None) -> None:
        """followups: {"mode": "off"|"observe"|"act", "model": the follow-up turn's model,
        "policy": "remote"|"local" (remote when there's an API key), "share_outcomes": bool
        (default True with an API key), "quiet_minutes": how long a user's silence takes to
        close an episode (30), "decide"/"send": your own decision and outcome functions}.
        api_key defaults to $XYBERNETEX_API_KEY, endpoint to $XYBERNETEX_ENDPOINT or
        https://api.xybernetex.com."""
        self._log_target = log
        self._approvals = approvals
        self._followups = dict(followups or {})
        if self._followups.get("mode", "observe") not in ("off", "observe", "act"):
            raise ValueError("followups.mode must be off, observe or act")
        if self._followups.get("policy", "remote") not in ("remote", "local"):
            raise ValueError("followups.policy must be remote or local")
        quiet = self._followups.get("quiet_minutes", 30)
        if not isinstance(quiet, (int, float)) or not 1 <= quiet <= 1440:
            raise ValueError("followups.quiet_minutes must be 1-1440")
        self._agent_id = agent_id
        key = api_key or os.environ.get("XYBERNETEX_API_KEY") or None
        endpoint = endpoint or os.environ.get("XYBERNETEX_ENDPOINT") or DEFAULT_ENDPOINT
        self._followup_mode = self._followups.get("mode", "observe") if followups else "off"
        remote = bool(key) and self._followups.get("policy", "remote") == "remote"
        self._decide_followup = self._followups.get("decide") or (RemoteDecider(key, endpoint) if remote else decide_local)
        send = self._followups.get("send") or (
            OutcomeSender(key, endpoint) if key and self._followups.get("share_outcomes", True) else None)
        self._outcomes = (OutcomeTracker(log=self._write, send=send, quiet_s=quiet * 60)
                          if self._followup_mode != "off" else None)
        self._authz = AuthorizationTracker()
        self._gate = ToolGate(mode=mode, preset=preset, rules=rules, log=self._write, approvals=approvals,
                              authorize=lambda event, ctx: self._authz.label((ctx or {}).get("session_key"),
                                                                             event.get("tool_name"), event.get("params")),
                              requests_target=lambda ctx, target: self._authz.requests_target((ctx or {}).get("session_key"), target),
                              owns_files=lambda event, ctx: self._authz.owns_files((ctx or {}).get("session_key"),
                                                                                   event.get("tool_name"), event.get("params")))
        self._decisions: dict[str, dict] = {}   # tool_call_id -> the gate's decision for that call
        self.rule_ids = self._gate.rule_ids
        self._write({"type": "tool_gate_ready", "mode": mode, "preset": preset or "none", "ruleIds": self.rule_ids,
                     "approvals": approvals, "followups": self._followup_mode,
                     **({"policy": "custom" if self._followups.get("decide") else "remote" if remote else "local",
                         "shareOutcomes": send is not None} if self._outcomes else {})})

    # ---- logging -------------------------------------------------------------------------

    def _write(self, entry: dict) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z", **entry}
        try:
            if callable(self._log_target):
                self._log_target(record)
                return
            path = Path(self._log_target) if self._log_target else DEFAULT_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break the agent
            pass

    # ---- the user's own words ----------------------------------------------------------

    def note_user_message(self, session_key: str, text: str) -> None:
        """Record a user turn for hooks-only use (run() does this itself)."""
        if not is_ours(text):
            self._authz.set_request(session_key, text, None)
            if self._outcomes:
                self._outcomes.note_user_turn(session_key, text)

    def end_session(self, session_key: str) -> None:
        """The conversation is over: close its episode and forget its requests."""
        self._authz.end_session(session_key)
        if self._outcomes:
            self._outcomes.end_session(session_key)

    def flush(self, timeout: float = 5.0) -> None:
        """Close every open episode and wait (up to `timeout` each) for outcome sends. Call before exit."""
        if self._outcomes:
            self._outcomes.flush(timeout)

    # ---- the gate on every tool call ---------------------------------------------------

    def _gate_decision(self, tool_name: str, args: Any, call_id: str, ctx: dict) -> dict:
        """The gate's decision for one call, made once and remembered by call id."""
        cached = self._decisions.get(call_id)
        if cached is not None:
            return cached
        params = args if isinstance(args, dict) else {}
        decision = self._gate({"tool_name": tool_name, "params": params, "tool_call_id": call_id}, ctx) or {}
        decision["_tool"] = tool_name
        decision["_params"] = params
        self._decisions[call_id] = decision
        while len(self._decisions) > 1000:
            del self._decisions[next(iter(self._decisions))]
        return decision

    def _resolve_hold(self, call_id: str, decision: str) -> None:
        """A person answered the hold on this call ("allow-once" or "deny")."""
        d = self._decisions.get(call_id) or {}
        hold = d.get("require_approval")
        if hold and not d.get("_resolved"):
            d["_resolved"] = True
            try:
                hold["on_resolution"](decision)
            except Exception:  # noqa: BLE001
                pass

    def _tool_done(self, tool_name: str, args: Any, call_id: str, output: Any, ctx: dict) -> None:
        """A tool call ran: what it created, and any delete command its output
        tells the agent to run (held as planted)."""
        self._resolve_hold(call_id, "allow-once")  # it ran, so a person approved any hold
        params = args if isinstance(args, dict) else {}
        try:
            self._authz.record_completed(ctx.get("session_key"), tool_name, params, False)
            self._gate.note_tool_result({"tool_name": tool_name, "params": params, "tool_call_id": call_id,
                                         "result": output}, ctx)
        except Exception:  # noqa: BLE001 - bookkeeping must never fail a tool
            pass

    def _work(self, calls: list[tuple[str | None, str | None]]) -> dict:
        """(tool name, call id) pairs -> tool calls, file writes and calls the gate refused, for an outcome."""
        refused = [cid for _, cid in calls if (self._decisions.get(cid) or {}).get("block")]
        writes = sum(1 for name, cid in calls if WRITE_TOOLS.match(name or "") and cid not in refused)
        return {"toolCalls": len(calls), "writes": writes, "failedCalls": len(refused)}

    # ---- after a run: the decision and its episode ------------------------------------

    def _run_end(self, session_key: str, agent_id: str | None, report: Report) -> None:
        self._write({"type": "run_end", "sessionKey": session_key, "agentId": agent_id,
                     "success": report.summary.success, "error": report.error, "toolCalls": report.summary.tool_calls,
                     "seconds": report.seconds, "held": report.status == "held"})

    async def _decide_after(self, session_key: str, agent_id: str | None, report: Report) -> dict | None:
        """The follow-up decision for an ended run (None when follow-ups are off or it's held)."""
        if self._followup_mode == "off" or report.status == "held":
            return None
        try:
            decision = await asyncio.to_thread(self._decide_followup, report.summary)
        except Exception as err:  # noqa: BLE001 - a custom decide() failing means no follow-up
            decision = {"action": "none", "probability": 1, "rule": f"decide-failed: {str(err)[:80]}"}
        decision.setdefault("policy", None)
        self._write({"type": "intervention", "sessionKey": session_key, "agentId": agent_id, "mode": self._followup_mode,
                     "model": report.summary.model, **decision})
        return decision

    def _followup_end(self, session_key: str, action: str, model: str | None, second: Report) -> None:
        self._write({"type": "followup_end", "sessionKey": session_key, "action": action, "model": model,
                     "success": second.summary.success, "toolCalls": second.summary.tool_calls, "seconds": second.seconds})

    def _episode(self, session_key: str, agent_id: str | None, report: Report, run_tokens: int | None,
                 second: Report | None, second_work: dict | None, second_tokens: int | None) -> None:
        if not self._outcomes or report.decision is None:
            return
        action = report.decision["action"]
        self._outcomes.open(
            session_key, agent_id=agent_id, model=report.summary.model, mode=self._followup_mode, decision=report.decision,
            applied=action if second is not None else "none",
            run={"success": report.summary.success, "retriable": report.summary.retriable,
                 "toolCalls": report.summary.tool_calls, "tokens": run_tokens},
            followup=None if second is None else {"success": second.status == "done", **(second_work or {}),
                                                  "tokens": second_tokens})


def model_label(model: Any) -> str | None:
    """A model's id: the string itself, or a model object's model name."""
    if isinstance(model, str):
        return model or None
    for attr in ("model", "model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None
