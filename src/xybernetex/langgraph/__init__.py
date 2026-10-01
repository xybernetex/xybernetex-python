"""Xybernetex for LangGraph.

    from xybernetex.langgraph import Xybernetex

    xyb = Xybernetex(mode="enforce", preset="recommended", followups={"mode": "act"})
    graph = create_react_agent(model, tools=xyb.tool_node(tools), checkpointer=InMemorySaver())
    report = await xyb.run(graph, "Delete the build folder", config={"configurable": {"thread_id": "chat-42"}})
    report.result["messages"]            # the graph's output, as usual
    report.followup                      # the follow-up turn's output, if one ran

    if report.status == "held":          # a destructive call nobody asked for
        report = await xyb.resume(graph, "allow-once", config=...)   # or "deny"

What it does, per run:

- **Safety Gate.** The gate sits in LangGraph's own tool node as a
  `wrap_tool_call` interceptor, so it sees every tool call of a prebuilt
  ReAct agent or a custom graph (`xyb.tool_node(tools)`, or pass
  `xyb.wrap_tool_call` / `xyb.awrap_tool_call` to your own `ToolNode`). A
  block reaches the model as the tool's result, with the reason. A hold is a
  LangGraph `interrupt()`: the run pauses (it needs a checkpointer), the app
  shows the interrupt's title and description, and resumes with
  `Command(resume="allow-once")` or `"deny"` (`xyb.resume` does both).
  With `approvals=False` (headless agents), a hold is a block that says why.
- **Run outcomes.** A run whose last reply is empty, or was cut off at the
  model's output limit (LangChain keeps the finish reason), is a death; so
  is a model API error.
- **Follow-ups.** After the run, the policy service (with an API key) or the
  local rule decides whether it gets one more turn - a retry when it died,
  a check-your-work turn when it finished. `mode` "act" runs it on
  `followup_graph` (a graph built with a stronger model, say) or the same
  graph, from the run's messages, on a thread of its own.
- **Outcomes.** As in the other adapters: episodes closed by the user's next
  message, logged and, with an API key, sent as labels and counts.

The user's request is the text passed to `run`, or - for a graph run
without `run` - the latest human message in the graph's state; never tool
output and never our own follow-up prompts. The log holds tool names, hashes
and labels, never prompts, files or command text.
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .._adapter import AdapterBase, ContractMixin, Report, call_hook, model_label
from ..core.contracts import Contract, contract_prompt
from ..core.deaths import retriable
from ..core.followups import MESSAGES, RunSummary, is_ours

__all__ = ["Xybernetex", "Report"]

_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("xybernetex_lg_session", default=None)
_FOLLOWUP_THREAD = ":xyb-"
_ALLOW = {"allow-once", "allow", "approve", "approved", "yes", "y"}
DENIED = ("Denied by the user: this call was not run. Don't retry it or get the same effect another way; "
          "tell the user what you wanted to do and why, and continue with the rest of the task.")


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c if isinstance(c, str) else str(c.get("text") or "") for c in content
                       if isinstance(c, (str, dict)) and (isinstance(c, str) or c.get("type") in (None, "text")))
    return ""


def _messages(state: Any) -> list:
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    if isinstance(state, list):
        return state
    return list(getattr(state, "messages", None) or [])


def _answer(value: Any) -> str:
    """A resume value -> "allow-once" or "deny"."""
    if isinstance(value, dict):
        value = value.get("decision", value.get("answer"))
    if value is True:
        return "allow-once"
    return "allow-once" if isinstance(value, str) and value.strip().lower() in _ALLOW else "deny"


class Xybernetex(AdapterBase, ContractMixin):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_user: dict[str, str] = {}  # session -> the last user text noted (so state reads don't repeat it)

    # ---- the gate in the tool node -----------------------------------------------------

    def tool_node(self, tools: Sequence[Any], **kwargs: Any) -> ToolNode:
        """A ToolNode for these tools with the gate on every call."""
        return ToolNode(tools, wrap_tool_call=self.wrap_tool_call, awrap_tool_call=self.awrap_tool_call, **kwargs)

    def _session(self, request: Any) -> str:
        own = _session_var.get()
        if own:
            return own
        config = getattr(getattr(request, "runtime", None), "config", None) or {}
        thread = str((config.get("configurable") or {}).get("thread_id") or "default")
        return thread.split(_FOLLOWUP_THREAD, 1)[0]

    def _note(self, session: str, text: str) -> None:
        self._last_user[session] = text
        while len(self._last_user) > 500:
            del self._last_user[next(iter(self._last_user))]
        self.note_user_message(session, text)

    def _note_from_state(self, session: str, state: Any) -> None:
        """For graphs run without `run`: the latest human message that isn't ours."""
        for m in reversed(_messages(state)):
            if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
                text = _text(m)
                if not is_ours(text):
                    if self._last_user.get(session) != text:
                        self._note(session, text)
                    return

    def _before(self, request: Any) -> tuple[Any, dict, str, dict, str]:
        tc = request.tool_call
        name, args, call_id = tc.get("name"), tc.get("args") or {}, tc.get("id") or ""
        session = self._session(request)
        try:
            self._note_from_state(session, request.state)
        except Exception:  # noqa: BLE001 - the request is best-effort context
            pass
        ctx = {"agent_id": self._agent_id, "session_key": session, "tool_call_id": call_id}
        return name, args, call_id, ctx, self._gate_decision(name, args, call_id, ctx)

    def _refusal(self, name: str, call_id: str, decision: dict) -> ToolMessage | None:
        """The block or the denied hold, as the tool's result; None when the call may run."""
        if decision.get("block"):
            return ToolMessage(content=decision["block_reason"], tool_call_id=call_id, name=name, status="error")
        hold = decision.get("require_approval")
        if hold:
            answer = _answer(interrupt({
                "type": "xybernetex_approval", "title": hold["title"], "description": hold["description"],
                "severity": hold["severity"], "allowed_decisions": hold["allowed_decisions"],
                "tool": name, "tool_call_id": call_id}))
            self._resolve_hold(call_id, answer)
            if answer != "allow-once":
                return ToolMessage(content=hold.get("timeout_reason") or DENIED, tool_call_id=call_id, name=name,
                                   status="error")
        return None

    def _after(self, name: str, args: dict, call_id: str, ctx: dict, result: Any) -> None:
        if isinstance(result, ToolMessage) and result.status != "error":
            self._tool_done(name, args, call_id, result.content, ctx)
        else:
            self._resolve_hold(call_id, "allow-once")

    def _governed(self, request: Any) -> ToolMessage | None:
        """The governor's stop, as the tool's result; None when the call may go on to the gate."""
        if self._governor is None:
            return None
        tc = request.tool_call
        turn = _this_turn(request.state)
        tokens = sum(int((m.usage_metadata or {}).get("total_tokens") or 0) for m in turn
                     if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None))
        stop = self._govern(self._session(request), tc.get("id") or "", tc.get("name"), tc.get("args") or {}, tokens)
        return None if stop is None else ToolMessage(content=stop, tool_call_id=tc.get("id") or "", name=tc.get("name"),
                                                     status="error")

    def wrap_tool_call(self, request: Any, execute: Any) -> Any:
        """ToolNode(wrap_tool_call=...): the governor and the gate before, the bookkeeping after."""
        stopped = self._governed(request)
        if stopped is not None:
            return stopped
        name, args, call_id, ctx, decision = self._before(request)
        refused = self._refusal(name, call_id, decision)
        if refused is not None:
            return refused
        result = execute(request)
        self._after(name, args, call_id, ctx, result)
        return result

    async def awrap_tool_call(self, request: Any, execute: Any) -> Any:
        """ToolNode(awrap_tool_call=...): the same, for async graphs."""
        stopped = self._governed(request)
        if stopped is not None:
            return stopped
        name, args, call_id, ctx, decision = self._before(request)
        refused = self._refusal(name, call_id, decision)
        if refused is not None:
            return refused
        result = await execute(request)
        self._after(name, args, call_id, ctx, result)
        return result

    # ---- running, with outcomes and follow-ups ------------------------------------------

    @staticmethod
    def _as_input(run_input: Any) -> Any:
        if isinstance(run_input, str):
            return {"messages": [HumanMessage(run_input)]}
        if isinstance(run_input, list):
            return {"messages": run_input}
        return run_input

    async def run(self, graph: Any, run_input: Any, *, config: dict | None = None, session_key: str | None = None,
                  followup_graph: Any = None, model: Any = None, contract: Any = None, run_checks: Any = None,
                  contract_model: Any = None, max_fixes: int = 1, after_first: Any = None, snapshots: Any = None) -> Report:
        """graph.ainvoke with the gate on, then outcome bookkeeping and the follow-up decision.
        `model` labels the run's model for decisions and outcomes (a model object or id).

        contract: a developer contract (core.contracts.Contract, or {"checks": [...]}) or "auto", written by
        contract_model (a LangChain chat model) while the graph runs. With one, the follow-up is decided by
        the checks, run through run_checks(command, timeout) -> (exit code, output) when the run ends: all
        pass -> done; any fail -> up to max_fixes targeted fix turns, each re-checked.

        after_first(report), sync or async, runs right after the first turn - before any check or follow-up
        touches the workspace - e.g. to grade or snapshot what the first turn alone produced.

        snapshots (snapshot() -> token, restore(token), discard(token); see core/ratchet.py) puts fix turns
        under the ratchet: a fix that makes a passing check fail is undone, and the agent is told."""
        if contract is not None and run_checks is None:
            raise ValueError("a contract needs run_checks=(command, timeout) -> (exit code, output)")
        config = dict(config or {})
        configurable = dict(config.get("configurable") or {})
        session = session_key or str(configurable.get("thread_id") or "default")
        token = _session_var.set(session)
        try:
            inp = self._as_input(run_input)
            text = None
            for m in reversed(_messages(inp)):
                if isinstance(m, HumanMessage) or (isinstance(m, tuple) and m[0] in ("user", "human")):
                    text = _text(m[1]) if isinstance(m, tuple) else _text(m)
                    if not is_ours(text):
                        self._note(session, text)
                    break
            fixed = self._contract_from(contract)
            writing = (asyncio.create_task(self._write_contract(contract_model, text))
                       if contract == "auto" and text is not None and contract_model is not None else None)
            self._governed_start(session)
            report = await self._one_turn(graph, inp, config)
            self._turn_done(session, _tokens(report.result))
            report.summary.model = model_label(model) if model is not None else None
            agent_id = self._agent_id
            self._run_end(session, agent_id, report)
            await call_hook(after_first, report)
            if contract is not None:
                error = None
                if writing is not None:
                    fixed, error, report.contract_tokens = await writing
                elif contract == "auto":
                    error = "contract='auto' needs contract_model" if contract_model is None else "no user text"
                self._contract_ready(session, fixed, error)
            report.contract = fixed
            verdict = None
            stop = self._stopped(session)
            if fixed is not None and report.status != "held":
                verdict = await self._check_contract(session, fixed, run_checks, 0)
                report.verdicts.append(verdict)
                report.final_verdict = verdict
            if stop and report.status != "held":
                report.decision = self._stop_decision(session, agent_id, report, stop)
            elif verdict is not None:
                report.decision = self._contract_decision(session, agent_id, report, verdict)
            else:
                report.decision = await self._decide_after(session, agent_id, report)
            if report.decision is None:
                return report
            second = None
            action = report.decision["action"]
            fixing = verdict is not None and not verdict.passed
            if self._followup_mode == "act" and action in MESSAGES:
                history = {"messages": _messages(report.result) if report.result is not None else _messages(inp)}
                target = followup_graph or self._followups.get("graph") or graph
                label = model_label(self._followups.get("model")) or report.summary.model

                async def turn(message: str) -> Report:
                    follow_config = {**config, "configurable": {
                        **configurable, "thread_id": f"{session}{_FOLLOWUP_THREAD}{action}-{time.time_ns()}"}}
                    rep = await self._one_turn(target, {"messages": [*history["messages"], HumanMessage(message)]},
                                               follow_config)
                    report.followup = rep.result
                    self._turn_done(session, _tokens(rep.result))
                    self._followup_end(session, action, label, rep)
                    if rep.result is not None:
                        history["messages"] = _messages(rep.result)
                    return rep
                second = (await self._fix_rounds(session, report, fixed, run_checks, verdict, max_fixes, snapshots, turn)
                          if fixing else await turn(MESSAGES[action]))
            self._episode(session, agent_id, report, _tokens(report.result), second,
                          self._work(_calls(second.result)) if second else None, _tokens(second.result) if second else None)
            return report
        finally:
            _session_var.reset(token)

    async def _write_contract(self, chat_model: Any, text: str) -> tuple[Contract | None, str | None, int | None]:
        """One call on a LangChain chat model: the request -> a generated contract."""
        try:
            reply = await chat_model.ainvoke([HumanMessage(contract_prompt(text))])
        except Exception as e:  # noqa: BLE001 - no contract: the run is decided the usual way
            return None, f"{type(e).__name__}: {e}"[:200], None
        parsed, error = self._parse_reply(_text(reply))
        usage = getattr(reply, "usage_metadata", None) or {}
        return parsed, error, int(usage.get("total_tokens") or 0) or None

    async def resume(self, graph: Any, answer: Any, *, config: dict, session_key: str | None = None) -> Report:
        """Answer a held call ("allow-once" / "deny", or True / False) and continue the run."""
        session = session_key or str((config.get("configurable") or {}).get("thread_id") or "default")
        token = _session_var.set(session)
        try:
            return await self._one_turn(graph, Command(resume=answer), config)
        finally:
            _session_var.reset(token)

    def run_sync(self, graph: Any, run_input: Any, **kwargs: Any) -> Report:
        """`run` for synchronous code (not from inside a running event loop)."""
        return asyncio.run(self.run(graph, run_input, **kwargs))

    async def _one_turn(self, graph: Any, graph_input: Any, config: dict) -> Report:
        t0 = time.time()
        try:
            out = await graph.ainvoke(graph_input, config)
        except GraphRecursionError as e:
            return Report(status="failed", error=f"recursion limit: {e}"[:200], seconds=round(time.time() - t0, 1),
                          summary=RunSummary(success=False, tool_calls=0, retriable=False, error="recursion limit"))
        except Exception as e:  # noqa: BLE001 - a model API error (timeout, overload) is a death too
            error = f"{type(e).__name__}: {e}"[:200]
            return Report(status="failed", error=error, seconds=round(time.time() - t0, 1),
                          summary=RunSummary(success=False, tool_calls=0, retriable=retriable(error), error=error))
        seconds = round(time.time() - t0, 1)
        turn = _this_turn(out)
        tool_calls = sum(1 for m in turn if isinstance(m, ToolMessage))
        interrupts = list(out.get("__interrupt__") or []) if isinstance(out, dict) else []
        if interrupts:
            return Report(result=out, status="held", seconds=seconds, interruptions=interrupts,
                          summary=RunSummary(success=False, tool_calls=tool_calls, retriable=False, error="held for approval"))
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
        error = None
        if final is not None and _finish_reason(final) in ("length", "max_tokens"):
            error = "no final answer: the model's reply was cut off at its output limit"
        elif final is None or (not _text(final).strip() and not final.tool_calls):
            error = "empty response from the model"
        if error:
            return Report(result=out, status="died", error=error, seconds=seconds,
                          summary=RunSummary(success=False, tool_calls=tool_calls, retriable=True, error=error))
        return Report(result=out, status="done", seconds=seconds, summary=RunSummary(success=True, tool_calls=tool_calls))


def _this_turn(out: Any) -> list[BaseMessage]:
    """The messages after the latest human message: what this run added."""
    messages = _messages(out)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def _finish_reason(message: AIMessage) -> str | None:
    meta = getattr(message, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason")
    return str(reason).lower() if reason else None


def _calls(out: Any) -> list[tuple[str | None, str | None]]:
    return [(tc.get("name"), tc.get("id")) for m in _this_turn(out) if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]


def _tokens(out: Any) -> int | None:
    total = [int((m.usage_metadata or {}).get("total_tokens") or 0) for m in _this_turn(out)
             if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None)]
    return sum(total) if total else None
