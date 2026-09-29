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
                 held targets, death detection, the policy client
  openai_agents/ the adapter for the OpenAI Agents SDK (hooks + tool wrappers)
  langgraph/     the adapter for LangGraph (later)
tests/           unittest; the JavaScript plugin's tests are the reference
                 cases, translated so both ports behave identically
```

## Status

Scaffold only. Nothing is ported yet. The order of work:

1. `core/risk.py` and `core/authz.py` - the classifier and the who-asked
   labels, with the plugin's tests translated.
2. `core/deaths.py` - empty and cut-off answers.
3. `openai_agents/` - a hook that gates tool calls and reports run ends.
4. The policy client and follow-up turns.

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
