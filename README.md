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
the model why and to ask the user. The log (default
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
| `core/authz.py` | who asked; held and planted targets | 27 | 1,402 labels, 1,402 result scans |
| `core/deaths.py` | deaths behind a reported success | 4 | 198 transcripts |
| `core/control.py` | the gate: rules, presets, holds, blocks | 33 | 1,402 decisions and log actions |
| `core/followups.py` | the follow-up prompts and local rule | - | same text as the plugin |
| `core/policy.py` | the policy-service client (`/intervene`, `/outcome`) | 6 | wire format = the plugin's; summaries pass the service's validator |
| `core/outcomes.py` | episodes: what happened after a decision | 8 | the plugin's classifier cases; episodes pass the service's validator |
| `openai_agents/` | the adapter | 10 end-to-end | scripted model through real `Runner.run` |

One deliberate difference from the plugin: an episode in act mode where the
rule held the follow-up out ("none", probability 0.1) records 0.1, the
chance it was actually left untreated; the plugin 0.4.1 records 1 there.

Not yet: the report and the LangGraph adapter. Known gap: the SDK drops the model's
finish reason, so a cut-off answer looks like a short answer; only an empty
answer is recognized as a death.

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
