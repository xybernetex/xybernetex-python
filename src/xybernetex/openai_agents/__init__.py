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
import time
from typing import Any, Callable

from agents import Agent, FunctionTool, Runner, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail
from agents.exceptions import MaxTurnsExceeded
from agents.items import ToolCallItem

from .._adapter import AdapterBase, ContractMixin, Report, call_hook, model_label
from ..core.contracts import Contract, contract_prompt
from ..core.deaths import retriable
from ..core.followups import MESSAGES, RunSummary

__all__ = ["Xybernetex", "Report"]

_session_var: contextvars.ContextVar[str] = contextvars.ContextVar("xybernetex_session", default="default")
_agent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("xybernetex_agent", default=None)


class Xybernetex(AdapterBase, ContractMixin):


    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._input_guardrail = ToolInputGuardrail(guardrail_function=self._on_tool_input, name="xybernetex-gate")
        self._output_guardrail = ToolOutputGuardrail(guardrail_function=self._on_tool_output, name="xybernetex-outcomes")

    # ---- the gate on every tool call ---------------------------------------------------

    def _ctx(self, agent_name: str | None, call_id: str | None) -> dict:
        # The approval callback isn't handed the agent, so run() leaves its name in a context var.
        return {"agent_id": self._agent_id or agent_name or _agent_var.get(), "session_key": _session_var.get(),
                "tool_call_id": call_id}

    def _decide(self, tool_name: str, args: Any, call_id: str, agent_name: str | None) -> dict:
        return self._gate_decision(tool_name, args, call_id, self._ctx(agent_name, call_id))

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
        self._tool_done(tc.tool_name, self._parse_args(tc), tc.tool_call_id, data.output,
                        self._ctx(getattr(data.agent, "name", None), tc.tool_call_id))
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
                  followup_model: Any = None, contract: Any = None, run_checks: Any = None, contract_model: Any = None,
                  max_fixes: int = 1, after_first: Any = None, snapshots: Any = None, **runner_kwargs: Any) -> Report:
        """Runner.run with the gate on, then outcome bookkeeping and the follow-up decision.

        contract: a developer contract (core.contracts.Contract, or {"checks": [...]}) or "auto" (written by
        contract_model, default the agent's own model, while the agent runs). With one, the follow-up is
        decided by the checks, run through run_checks(command, timeout) -> (exit code, output) when the run
        ends: all pass -> done; any fail -> up to max_fixes targeted fix turns, each re-checked.

        after_first(report), sync or async, runs right after the first turn - before any check or follow-up
        touches the workspace - e.g. to grade or snapshot what the first turn alone produced.

        snapshots (snapshot() -> token, restore(token), discard(token); see core/ratchet.py) puts fix turns
        under the ratchet: a fix that makes a passing check fail is undone, and the agent is told."""
        if contract is not None and run_checks is None:
            raise ValueError("a contract needs run_checks=(command, timeout) -> (exit code, output)")
        token = _session_var.set(session_key)
        agent_token = _agent_var.set(getattr(agent, "name", None))
        try:
            text = self._user_text(run_input)
            if text is not None:
                self.note_user_message(session_key, text)
            fixed = self._contract_from(contract)
            writing = (asyncio.create_task(self._write_contract(contract_model or agent.model, text))
                       if contract == "auto" and text is not None else None)
            guarded = self.guard(agent)
            report = await self._one_turn(guarded, run_input, max_turns, runner_kwargs)
            report.summary.model = model_label(agent.model)
            agent_id = self._agent_id or agent.name
            self._run_end(session_key, agent_id, report)
            await call_hook(after_first, report)
            if contract is not None:
                error = None
                if writing is not None:
                    fixed, error, report.contract_tokens = await writing
                elif contract == "auto":
                    error = "no user text to write a contract from"
                self._contract_ready(session_key, fixed, error)
            report.contract = fixed
            verdict = None
            if fixed is not None and report.status != "held":
                verdict = await self._check_contract(session_key, fixed, run_checks, 0)
                report.verdicts.append(verdict)
                report.final_verdict = verdict
                report.decision = self._contract_decision(session_key, agent_id, report, verdict)
            else:
                report.decision = await self._decide_after(session_key, agent_id, report)
            if report.decision is None:
                return report
            second = None
            action = report.decision["action"]
            fixing = verdict is not None and not verdict.passed
            if (self._followup_mode == "act" and action in MESSAGES
                    and (report.result is not None or action == "retry" or fixing)):
                model = followup_model or self._followups.get("model")
                follow_agent = guarded.clone(model=model) if model else guarded
                # A run that raised has no transcript: the follow-up starts from the
                # original input (the files it changed are still there).
                history = {"items": report.result.to_input_list() if report.result is not None
                           else [{"role": "user", "content": run_input}] if isinstance(run_input, str) else list(run_input)}
                label = model_label(model) if model else report.summary.model

                async def turn(message: str) -> Report:
                    rep = await self._one_turn(follow_agent, history["items"] + [{"role": "user", "content": message}],
                                               max_turns, runner_kwargs)
                    report.followup = rep.result
                    self._followup_end(session_key, action, label, rep)
                    if rep.result is not None:
                        history["items"] = rep.result.to_input_list()
                    return rep
                second = (await self._fix_rounds(session_key, report, fixed, run_checks, verdict, max_fixes, snapshots, turn)
                          if fixing else await turn(MESSAGES[action]))
            self._episode(session_key, agent_id, report, _tokens(report.result), second,
                          self._work(self._calls(second.result)) if second else None,
                          _tokens(second.result) if second else None)
            return report
        finally:
            _agent_var.reset(agent_token)
            _session_var.reset(token)

    async def _write_contract(self, model: Any, text: str) -> tuple[Contract | None, str | None, int | None]:
        """One call on the given model: the request -> a generated contract (no tools, one turn)."""
        writer = Agent(name="xybernetex-contract", instructions="You write acceptance checks. Reply with JSON only.",
                       model=model)
        try:
            result = await Runner.run(writer, contract_prompt(text), max_turns=1)
        except Exception as e:  # noqa: BLE001 - no contract: the run is decided the usual way
            return None, f"{type(e).__name__}: {e}"[:200], None
        parsed, error = self._parse_reply(result.final_output)
        return parsed, error, _tokens(result)

    async def resume(self, agent: Agent, state: Any, *, session_key: str = "default", **runner_kwargs: Any) -> Report:
        """Continue a run that paused for approval, after approve/reject on its RunState."""
        token = _session_var.set(session_key)
        agent_token = _agent_var.set(getattr(agent, "name", None))
        try:
            return await self._one_turn(self.guard(agent), state, None, runner_kwargs)
        finally:
            _agent_var.reset(agent_token)
            _session_var.reset(token)

    def _calls(self, result: Any) -> list[tuple[str | None, str | None]]:
        return [self._call_of(i)[:2] for i in (result.new_items if result is not None else []) if isinstance(i, ToolCallItem)]

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


def _tokens(result: Any) -> int | None:
    try:
        return int(result.context_wrapper.usage.total_tokens)
    except Exception:  # noqa: BLE001 - no result, or no usage reported
        return None
