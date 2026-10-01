# xybernetex (Python)

The Xybernetex control plane for Python agent frameworks: the OpenAI Agents
SDK first, LangGraph next. A port of what the OpenClaw plugin
(`xybernetex-openclaw`, JavaScript) does, with one shared core and one thin
adapter per framework.

## What it will do

The same three things as the OpenClaw plugin:

- **Safety Gate.** Classify each tool call's risk from what it visibly does
  (a shell command, a file path, a patch), label it with who asked for it
  (the user's own message, the agent's own cleanup, or nobody), and hold
  destructive actions nobody asked for. Instructions planted in files or web
  pages can't authorize themselves; the user's own requests run without a
  prompt. Runs locally, no API key.
- **Run outcomes.** Notice when a run died even though the framework reports
  success (an empty answer, a cut-off answer), and record what the agent did.
- **Follow-ups.** After a run ends, decide whether it gets one more turn - a
  retry when it died, a check-your-work turn when it finished, on the run's
  own model or a stronger one - either by a local rule or by asking the
  Xybernetex policy service, which sees tool names, hashes and labels, never
  prompts or files.

## Layout

```
src/xybernetex/
  core/          framework-independent: risk classification, who-asked labels,
                 held targets, death detection, follow-ups, outcomes, the policy client
  openai_agents/ the adapter for the OpenAI Agents SDK (hooks + tool wrappers)
  langgraph/     the adapter for LangGraph (later)
tests/           unittest; the JavaScript plugin's tests are the reference
                 cases, translated so both ports behave identically
```

## Usage (OpenAI Agents SDK)

```python
from xybernetex.openai_agents import Xybernetex

xyb = Xybernetex(mode="enforce", preset="recommended",
                 followups={"mode": "act", "model": stronger_model})   # or mode="observe"
report = await xyb.run(agent, "Delete the build folder", session_key="chat-42")
report.result.final_output   # the SDK's RunResult, as usual
report.status                # done | died | held | failed
report.followup              # the follow-up turn's RunResult, if one ran

if report.status == "held":                     # a destructive call nobody asked for
    state = report.result.to_state()
    state.approve(report.interruptions[0])      # or state.reject(..., rejection_message=...)
    report = await xyb.resume(agent, state, session_key="chat-42")

xyb.end_session("chat-42")   # when the conversation is over
xyb.flush()                  # before the process exits: closes open episodes, sends them
```

`approvals=False` is for headless agents: a hold becomes a block that tells
the model why and to ask the user. The agent may delete files it created itself
(a plain `rm`/`del`/`Remove-Item` of single files, nothing moved onto them
since) without either; folders it made stay held, except tool caches. The log (default
`~/.xybernetex/events.jsonl`) holds the same entries as the OpenClaw
plugin's: tool names, hashes and labels, never prompts, files or command
text.

**The policy service.** With an API key (`api_key=` or
`XYBERNETEX_API_KEY`), follow-up decisions come from api.xybernetex.com
(`POST /intervene`, told only success, retriable, tool-call count and model
id) and fall back to the local rule if it can't answer. Each decision opens an
episode that closes with what the user did next - a correction, the same
request again, thanks, something new, or silence (`quiet_minutes`, 30) - and
is sent as labels and counts (`POST /outcome`); the user's words are
classified in memory and never logged or sent. `followups={"policy":
"local"}` keeps decisions local; `"share_outcomes": False` keeps episodes in
the local log. Without a key, everything stays local.

One SDK behaviour the adapter covers for you: when a tool has a callable
`needs_approval`, the SDK consults it only if the model's arguments round-trip
through the tool's schema unchanged; a parameter with a default that the model
left out makes the SDK pause for approval on its own, without asking any
policy. The adapter decides those pauses the way the gate would have (allowed
calls resume by themselves, blocks are rejected with the reason, real holds
stay for a person), so tools with optional parameters work the same as any
other.

## Status

Ported and cross-checked against the JavaScript on real data (every
experiment run replayed through both implementations, identical results):

| Module | What | Tests | Parity check |
| --- | --- | --- | --- |
| `core/risk.py` | what a call does | 11 | 497 shell commands, 1,402 tool calls |
| `core/authz.py` | who asked; held and planted targets; own files | 27 + 11 | 1,402 labels, 1,402 result scans |
| `core/deaths.py` | deaths behind a reported success | 4 | 198 transcripts |
| `core/control.py` | the gate: rules, presets, holds, blocks | 33 | 1,402 decisions and log actions |
| `core/followups.py` | the follow-up prompts and local rule | - | same text as the plugin |
| `core/policy.py` | the policy-service client (`/intervene`, `/outcome`) | 6 | wire format = the plugin's; summaries pass the service's validator |
| `core/contracts.py` | acceptance checks: parse, refuse unsafe, run, the fix message | 10 + 8 end-to-end | - |
| `core/ratchet.py` | keep or undo each fix by which checks pass; folder snapshots | 5 | - |
| `core/governor.py` | budgets, repeats and no-progress stops | 7 + 4 end-to-end | - |
| `core/outcomes.py` | episodes: what happened after a decision | 8 | the plugin's classifier cases; episodes pass the service's validator |
| `openai_agents/` | the OpenAI Agents SDK adapter | 13 end-to-end | scripted model through real `Runner.run` |
| `langgraph/` | the LangGraph adapter | 11 end-to-end | scripted chat model through a real ReAct graph, interrupts and resumes |

One deliberate difference from the plugin: an episode in act mode where the
rule held the follow-up out ("none", probability 0.1) records 0.1, the
chance it was actually left untreated; the plugin 0.4.1 records 1 there.

Not yet: the report, and LangChain 1.0's `create_agent` middleware. Known gap: the SDK drops the model's
finish reason, so a cut-off answer looks like a short answer; only an empty
answer is recognized as a death.

## Usage (LangGraph)

```python
from xybernetex.langgraph import Xybernetex

xyb = Xybernetex(mode="enforce", preset="recommended", followups={"mode": "act"})
graph = create_react_agent(model, tools=xyb.tool_node(tools), checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "chat-42"}}
report = await xyb.run(graph, "Delete the build folder", config=config,
                       followup_graph=stronger_graph)   # optional: follow-ups on another model
if report.status == "held":                             # LangGraph interrupt(), with our title and reason
    report = await xyb.resume(graph, "allow-once", config=config)   # or "deny"
```

The gate is a `ToolNode` interceptor (`wrap_tool_call`), so it covers the
prebuilt ReAct agent and custom graphs alike: use `xyb.tool_node(tools)`,
or pass `xyb.wrap_tool_call` / `xyb.awrap_tool_call` to your own `ToolNode`.
A graph invoked directly, without `xyb.run`, is still gated; the user's
request is then read from the latest human message in the graph's state.
LangChain keeps each reply's finish reason, so here a reply cut off at the
output limit is recognized as a death, not just an empty one. Not covered
yet: LangChain 1.0's `create_agent` (it takes middleware instead of a
`ToolNode`).

## Contracts and the ratchet

A contract says what "done" means for a run, as checks code can run:

```python
report = await xyb.run(agent, "Write report.csv ...", session_key="s",
                       contract="auto",            # or {"checks": [{"name": ..., "command": ..., "expect": ...}]}
                       run_checks=run_in_workspace, # (command, timeout) -> (exit code, output)
                       snapshots=FolderSnapshots(workdir),  # optional: the ratchet
                       max_fixes=2)
report.contract_met      # do the checks pass on the workspace the run ended with?
report.ratchet           # per fix round: improved | same | regressed | rolled-back
```

`"auto"` has the agent's own model write the checks while the agent works;
checks that would change anything (deletes, redirects into files, installs,
network) are refused. When the run ends the checks run, costing no tokens:
all pass means done, with no check-your-work turn; any failure gets a fix
turn naming exactly what failed. With `snapshots`, every fix turn is under
the ratchet: a fix that makes a passing check fail is undone and the agent
is told, so a run's progress only goes up. `after_first=` lets you grade or
snapshot the first turn before anything else touches the workspace.

## The governor

```python
xyb = Xybernetex(mode="enforce", preset="recommended", governor="standard")
# or governor={"max_tool_calls": 100, "max_tokens": 500_000, "max_seconds": 1800,
#              "repeat_limit": 4, "no_progress_rounds": 2}
report.governor          # None, or why the run was stopped
```

Deterministic stops for runs that spend without progress, counted across
everything one `xyb.run` does (first turn, follow-ups, fix rounds): a budget
of tool calls, tokens or wall-clock time, the same call made several times
in a row, or fix rounds that keep failing to improve the contract. The call
that crosses a limit, and every call after it, is answered with a plain
"stop and summarize" instead of running, and no follow-up starts. Off by
default; `"standard"` is 150 calls, 1.5M tokens, an hour, 4 repeats, 2
rounds without progress.

## Development

Python 3.11 or newer.

```
pip install -e ".[openai-agents]"
python -m unittest discover -s tests
```

The experiment that runs our benchmark tasks through the SDK lives in the
trainer repo (`xybernetex-trainer/scenarios/sdk_agent.py`), because it needs
the grader there; it will import this package as the port fills in.

## Related repos

- `xybernetex-openclaw` - the OpenClaw plugin this ports.
- `xybernetex-cfworker` - the policy service (api.xybernetex.com).
- `xybernetex-trainer` - experiments, graders and the learner.
