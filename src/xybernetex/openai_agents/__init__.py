"""Xybernetex for the OpenAI Agents SDK.

    from xybernetex.openai_agents import Xybernetex

    xyb = Xybernetex(mode="enforce", preset="recommended", followups={"mode": "act", "model": "stronger-model"})
    report = await xyb.run(agent, "Delete the build folder", session_key="chat-42")
    report.result.final_output           # the SDK's RunResult, as usual
    report.followup                      # the follow-up turn's RunResult, if one ran

What it does, per run:

- **Safety Gate.** Every function tool on the agent is wrapped: its risk is
  classified from what it does (core/risk.py), labeled with who asked for it
  (core/authz.py) and judged by the gate (core/control.py). A block reaches
  the model as the tool's result, with the reason. A hold uses the SDK's own
  approval flow: the run pauses with `result.interruptions`, and the app
  approves or rejects on `result.to_state()` and resumes with `xyb.resume`.
  With `approvals=False` (headless agents), a hold is a block that says why.
- **Run outcomes.** A run that ended with nothing usable is treated as a
  death (core/deaths.py), whatever the SDK reported.
- **Follow-ups.** After the run, the policy service (with an API key) or
  the local rule (core/followups.py) decides whether it gets one more turn -
  a retry when it died, a check-your-work turn when it finished - in the
  same conversation, on the run's model or on `followups["model"]`. `mode`
  "observe" decides and logs but starts nothing; "act" starts the turn.
- **Outcomes.** Each decision opens an episode (core/outcomes.py) that
  closes with what the user did next - a correction, the same request again,
  thanks, something new, or nothing - and is logged and, with an API key,
  sent to the policy service as labels and counts (turn off with
  `followups["share_outcomes"] = False`). Call `end_session` when a
  conversation ends and `flush` before the process exits.

Only the user's own input counts as the user's request: the text passed to
`run` (or `note_user_message`), never tool output, and never our own
follow-up prompts. The log holds tool names, hashes and labels, never prompts,
files or command text.
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agents import Agent, FunctionTool, Runner, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail
from agents.exceptions import AgentsException, MaxTurnsExceeded
from agents.items import ToolCallItem

from ..core.authz import AuthorizationTracker
from ..core.control import ToolGate
from ..core.deaths import retriable
from ..core.followups import MESSAGES, RunSummary, decide_local, is_ours
from ..core.outcomes import WRITE_TOOLS, OutcomeTracker
from ..core.policy import DEFAULT_ENDPOINT, OutcomeSender, RemoteDecider

DEFAULT_LOG = Path.home() / ".xybernetex" / "events.jsonl"
_session_var: contextvars.ContextVar[str] = contextvars.ContextVar("xybernetex_session", default="default")
_agent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("xybernetex_agent", default=None)


@dataclass
class Report:
    """What `run` returns: the SDK's result plus what Xybernetex did with it."""
    result: Any = None                      # RunResult, or None when the run raised
    status: str = "done"                    # done | died | held | failed
    error: str | None = None
    summary: RunSummary | None = None
    decision: dict | None = None            # the follow-up decision, if follow-ups are configured
    followup: Any = None                    # the follow-up turn's RunResult, if one ran
    seconds: float = 0.0
    interruptions: list = field(default_factory=list)


class Xybernetex:
    def __init__(self, *, mode: str = "observe", preset: str | None = "recommended", rules: list[dict] | None = None,
                 log: Callable[[dict], Any] | str | os.PathLike | None = None, approvals: bool = True,
                 followups: dict | None = None, agent_id: str | None = None, api_key: str | None = None,
                 endpoint: str | None = None) -> None:
        """followups: {"mode": "off"|"observe"|"act", "model": a Model for the follow-up turn,
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
        self._input_guardrail = ToolInputGuardrail(guardrail_function=self._on_tool_input, name="xybernetex-gate")
        self._output_guardrail = ToolOutputGuardrail(guardrail_function=self._on_tool_output, name="xybernetex-outcomes")
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

    def _ctx(self, agent_name: str | None, call_id: str | None) -> dict:
        # The approval callback isn't handed the agent, so run() leaves its name in a context var.
        return {"agent_id": self._agent_id or agent_name or _agent_var.get(), "session_key": _session_var.get(),
                "tool_call_id": call_id}

    def _decide(self, tool_name: str, args: Any, call_id: str, agent_name: str | None) -> dict:
        """The gate's decision for one call, made once and remembered by call id."""
        cached = self._decisions.get(call_id)
        if cached is not None:
            return cached
        params = args if isinstance(args, dict) else {}
        decision = self._gate({"tool_name": tool_name, "params": params, "tool_call_id": call_id},
                              self._ctx(agent_name, call_id)) or {}
        decision["_tool"] = tool_name
        decision["_params"] = params
        self._decisions[call_id] = decision
        while len(self._decisions) > 1000:
            del self._decisions[next(iter(self._decisions))]
        return decision

    def _needs_approval_for(self, tool: FunctionTool) -> Callable:
        async def needs_approval(ctx: Any, args: dict, call_id: str) -> bool:
            agent_name = getattr(getattr(ctx, "agent", None), "name", None)
            return "require_approval" in self._decide(tool.name, args, call_id, agent_name)
        return needs_approval

    @staticmethod
    def _parse_args(tool_context: Any) -> Any:
        raw = getattr(tool_context, "tool_arguments", None)
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return {}

    def _on_tool_input(self, data: Any) -> ToolGuardrailFunctionOutput:
        tc = data.context
        decision = self._decide(tc.tool_name, self._parse_args(tc), tc.tool_call_id, getattr(data.agent, "name", None))
        if decision.get("block"):
            return ToolGuardrailFunctionOutput.reject_content(decision["block_reason"], output_info={"xybernetex": "blocked"})
        return ToolGuardrailFunctionOutput.allow()

    def _on_tool_output(self, data: Any) -> ToolGuardrailFunctionOutput:
        tc = data.context
        args = self._parse_args(tc)
        decision = self._decisions.get(tc.tool_call_id) or {}
        hold = decision.get("require_approval")
        if hold and not decision.get("_resolved"):
            # The tool ran, so a person approved it.
            decision["_resolved"] = True
            try:
                hold["on_resolution"]("allow-once")
            except Exception:  # noqa: BLE001
                pass
        ctx = self._ctx(getattr(data.agent, "name", None), tc.tool_call_id)
        try:
            self._authz.record_completed(ctx["session_key"], tc.tool_name, args, False)
            self._gate.note_tool_result({"tool_name": tc.tool_name, "params": args, "tool_call_id": tc.tool_call_id,
                                         "result": data.output}, ctx)
        except Exception:  # noqa: BLE001 - bookkeeping must never fail a tool
            pass
        return ToolGuardrailFunctionOutput.allow()

    def protect(self, tools: list) -> list:
        """The same tools with the gate on every function tool. Other tool kinds pass through."""
        out = []
        for tool in tools:
            if not isinstance(tool, FunctionTool) or self._input_guardrail in (tool.tool_input_guardrails or []):
                out.append(tool)
                continue
            out.append(dataclasses.replace(
                tool, needs_approval=self._needs_approval_for(tool),
                tool_input_guardrails=[*(tool.tool_input_guardrails or []), self._input_guardrail],
                tool_output_guardrails=[*(tool.tool_output_guardrails or []), self._output_guardrail]))
        return out

    def guard(self, agent: Agent) -> Agent:
        """A copy of the agent with its function tools protected."""
        return agent.clone(tools=self.protect(list(agent.tools)))

    # ---- running, with outcomes and follow-ups ------------------------------------------

    @staticmethod
    def _user_text(run_input: Any) -> str | None:
        if isinstance(run_input, str):
            return run_input
        if isinstance(run_input, list):
            for item in reversed(run_input):
                if isinstance(item, dict) and item.get("role") == "user":
                    content = item.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
        return None

    async def run(self, agent: Agent, run_input: Any, *, session_key: str = "default", max_turns: int | None = None,
                  followup_model: Any = None, **runner_kwargs: Any) -> Report:
        """Runner.run with the gate on, then outcome bookkeeping and the follow-up decision."""
        token = _session_var.set(session_key)
        agent_token = _agent_var.set(getattr(agent, "name", None))
        try:
            text = self._user_text(run_input)
            if text is not None:
                self.note_user_message(session_key, text)
            guarded = self.guard(agent)
            report = await self._one_turn(guarded, run_input, max_turns, runner_kwargs)
            report.summary.model = _model_label(agent.model)
            agent_id = self._agent_id or agent.name
            self._write({"type": "run_end", "sessionKey": session_key, "agentId": agent_id,
                         "success": report.summary.success, "error": report.error, "toolCalls": report.summary.tool_calls,
                         "seconds": report.seconds, "held": report.status == "held"})
            mode = self._followup_mode
            if mode == "off" or report.status == "held":
                return report
            try:
                report.decision = await asyncio.to_thread(self._decide_followup, report.summary)
            except Exception as err:  # noqa: BLE001 - a custom decide() failing means no follow-up
                report.decision = {"action": "none", "probability": 1, "rule": f"decide-failed: {str(err)[:80]}"}
            report.decision.setdefault("policy", None)
            self._write({"type": "intervention", "sessionKey": session_key, "agentId": agent_id, "mode": mode,
                         "model": report.summary.model, **report.decision})
            second = None
            action = report.decision["action"]
            if mode == "act" and action in MESSAGES and (report.result is not None or action == "retry"):
                model = followup_model or self._followups.get("model")
                follow_agent = guarded.clone(model=model) if model else guarded
                # A run that raised has no transcript: the retry starts from the
                # original input (the files it changed are still there).
                before = (report.result.to_input_list() if report.result is not None
                          else [{"role": "user", "content": run_input}] if isinstance(run_input, str) else list(run_input))
                history = before + [{"role": "user", "content": MESSAGES[action]}]
                second = await self._one_turn(follow_agent, history, max_turns, runner_kwargs)
                report.followup = second.result
                self._write({"type": "followup_end", "sessionKey": session_key, "action": action,
                             "model": _model_label(model) if model else report.summary.model,
                             "success": second.summary.success, "toolCalls": second.summary.tool_calls,
                             "seconds": second.seconds})
            if self._outcomes:
                self._outcomes.open(
                    session_key, agent_id=agent_id, model=report.summary.model, mode=mode, decision=report.decision,
                    applied=action if second is not None else "none",
                    run={"success": report.summary.success, "retriable": report.summary.retriable,
                         "toolCalls": report.summary.tool_calls, "tokens": _tokens(report.result)},
                    followup=None if second is None else {
                        "success": second.status == "done", **self._work(second.result), "tokens": _tokens(second.result)})
            return report
        finally:
            _agent_var.reset(agent_token)
            _session_var.reset(token)

    async def resume(self, agent: Agent, state: Any, *, session_key: str = "default", **runner_kwargs: Any) -> Report:
        """Continue a run that paused for approval, after approve/reject on its RunState."""
        token = _session_var.set(session_key)
        agent_token = _agent_var.set(getattr(agent, "name", None))
        try:
            return await self._one_turn(self.guard(agent), state, None, runner_kwargs)
        finally:
            _agent_var.reset(agent_token)
            _session_var.reset(token)

    def _work(self, result: Any) -> dict:
        """A run's tool calls, file writes and calls the gate refused, for its outcome."""
        calls = [self._call_of(i) for i in (result.new_items if result is not None else []) if isinstance(i, ToolCallItem)]
        refused = [cid for _, cid, _ in calls if (self._decisions.get(cid) or {}).get("block")]
        writes = sum(1 for name, cid, _ in calls if WRITE_TOOLS.match(name or "") and cid not in refused)
        return {"toolCalls": len(calls), "writes": writes, "failedCalls": len(refused)}

    @staticmethod
    def _call_of(item: Any) -> tuple[str | None, str | None, Any]:
        raw = item.raw_item
        get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
        return get("name"), get("call_id"), get("arguments")

    async def _settle_sdk_holds(self, agent: Agent, result: Any, kwargs: dict) -> Any:
        """Decide the holds the SDK raised on its own.

        When a tool has a callable needs_approval, the SDK only consults it if
        the model's arguments round-trip through the tool's schema unchanged;
        otherwise (a parameter with a default the model left out, a coerced
        value) it pauses for approval without asking anyone. Those pauses carry
        no decision of ours, so they'd reach the host as holds with no reason.
        Here the gate decides them the way needs_approval would have: allowed
        calls are approved and the run resumes (the input guardrail re-checks
        the same cached decision at execution), blocks are rejected with the
        gate's reason, and a genuine hold stays for a person to decide."""
        for _ in range(50):  # each pass resumes once; a run pausing this often is not settling
            pending = list(getattr(result, "interruptions", None) or [])
            state, settled = None, 0
            for item in pending:
                name, call_id, arguments = self._call_of(item)
                if not name or not call_id or call_id in self._decisions:
                    continue  # ours (the gate already decided, a person decides next), or not a function call
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})
                except ValueError:
                    args = {}
                decision = self._decide(name, args, call_id, getattr(agent, "name", None))
                if decision.get("require_approval"):
                    continue  # now a hold with a reason, left for the host
                state = state or result.to_state()
                if decision.get("block"):
                    state.reject(item, rejection_message=decision["block_reason"])
                else:
                    state.approve(item)
                settled += 1
            if not settled:
                return result
            result = await Runner.run(agent, state, **kwargs)
        return result

    async def _one_turn(self, agent: Agent, run_input: Any, max_turns: int | None, runner_kwargs: dict) -> Report:
        t0 = time.time()
        kwargs = dict(runner_kwargs)
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        try:
            result = await Runner.run(agent, run_input, **kwargs)
            result = await self._settle_sdk_holds(agent, result, kwargs)
        except MaxTurnsExceeded as e:
            return Report(status="failed", error=f"max turns exceeded: {e}"[:200], seconds=round(time.time() - t0, 1),
                          summary=RunSummary(success=False, tool_calls=0, retriable=False, error="max turns exceeded"))
        except Exception as e:  # noqa: BLE001
            # The SDK's own failures, and the model API's (a 408 request timeout,
            # an overloaded provider) that the SDK passes through as they are:
            # the run died, and a timeout is the death a retry is for.
            error = f"{type(e).__name__}: {e}"[:200]
            return Report(status="failed", error=error, seconds=round(time.time() - t0, 1),
                          summary=RunSummary(success=False, tool_calls=0, retriable=retriable(error), error=error))
        seconds = round(time.time() - t0, 1)
        tool_calls = sum(1 for item in result.new_items if isinstance(item, ToolCallItem))
        if getattr(result, "interruptions", None):
            return Report(result=result, status="held", seconds=seconds, interruptions=list(result.interruptions),
                          summary=RunSummary(success=False, tool_calls=tool_calls, retriable=False, error="held for approval"))
        final = result.final_output
        # The SDK reports success whenever the model call didn't fail; an empty
        # final answer after a run is the death the OpenClaw plugin learned to see.
        if final is None or (isinstance(final, str) and not final.strip()):
            error = "empty response from the model"
            return Report(result=result, status="died", error=error, seconds=seconds,
                          summary=RunSummary(success=False, tool_calls=tool_calls, retriable=True, error=error))
        return Report(result=result, status="done", seconds=seconds, summary=RunSummary(success=True, tool_calls=tool_calls))


def _model_label(model: Any) -> str | None:
    """The model's id: the string itself, or a Model object's model name."""
    if isinstance(model, str):
        return model or None
    for attr in ("model", "model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _tokens(result: Any) -> int | None:
    try:
        return int(result.context_wrapper.usage.total_tokens)
    except Exception:  # noqa: BLE001 - no result, or no usage reported
        return None
