"""Contracts through both adapters, end to end with scripted models: the
checks decide the follow-up - met means done (even after a reported death),
failed means a targeted fix turn and a re-check - for developer contracts and
generated ones alike."""
import asyncio
import json
import unittest
import warnings

warnings.filterwarnings("ignore")
try:
    import test_langgraph as lgt
    import test_openai_agents as sdkt
except unittest.SkipTest:  # pragma: no cover
    raise

from xybernetex.core.followups import MARKER

CONTRACT = {"checks": [{"name": "report exists", "command": "test -f report.csv"},
                       {"name": "tests pass", "command": "python3 -m pytest -q"}]}
GENERATED = json.dumps({"checks": [{"name": "report exists", "command": "test -f report.csv"},
                                   {"name": "wipe it", "command": "rm -rf build"}]})


class Workspace:
    """A fake executor over a pretend folder: `fixed` flips once the agent runs the fix."""

    def __init__(self, ran, fail_until=None):
        self.ran, self.fail_until, self.seen = ran, fail_until, []

    def __call__(self, command, timeout):
        self.seen.append(command)
        if command == "python3 -m pytest -q" and self.fail_until is not None and self.fail_until not in self.ran:
            return 1, "1 failed\nFAILED test_report.py::test_totals"
        return 0, "ok"


def last_user_text(framework, h, model):
    if framework == "sdk":
        sent = model.inputs[-1]
        return next(str(i["content"]) for i in reversed(sent) if isinstance(i, dict) and i.get("role") == "user")
    return next(m.content for m in reversed(model.seen[-1]) if isinstance(m, lgt.HumanMessage))


class ContractRunTest(unittest.TestCase):
    def harness(self, framework, *turns, **xyb):
        if framework == "sdk":
            h = sdkt.Harness(*[t if isinstance(t, list) else [t] for t in turns], **xyb)
            return h, h.model
        h = lgt.Harness(*turns, **xyb)
        return h, h.model

    def steps(self, framework):
        mod = sdkt if framework == "sdk" else lgt
        return mod.call, mod.say

    def test_a_met_contract_ends_the_run_with_no_follow_up(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    followups={"mode": "act"})
            ws = Workspace(h.ran)
            report = h.run("Write report.csv and make the tests pass.", contract=CONTRACT, run_checks=ws)
            self.assertEqual((report.contract_met, report.decision["rule"], report.decision["policy"]),
                             (True, "contract-met", "contract"), fw)
            self.assertIsNone(report.followup, fw)
            self.assertEqual(ws.seen, ["test -f report.csv", "python3 -m pytest -q"], fw)
            kinds = [e["type"] for e in h.logs]
            self.assertIn("contract", kinds)
            self.assertIn("contract_check", kinds)
            self.assertNotIn("report.csv", json.dumps([e for e in h.logs if e["type"].startswith("contract")]), fw)

    def test_a_failed_contract_gets_a_targeted_fix_and_a_recheck(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    call("run_command", {"command": "python3 fix_totals.py"}, "c2"), say("Fixed the totals."),
                                    followups={"mode": "act"})
            ws = Workspace(h.ran, fail_until="python3 fix_totals.py")
            report = h.run("Write report.csv and make the tests pass.", contract=CONTRACT, run_checks=ws)
            self.assertEqual((report.decision["rule"], len(report.verdicts), report.contract_met), ("contract-failed", 2, True), fw)
            sent = last_user_text(fw, h, model)
            self.assertTrue(sent.startswith(MARKER), fw)
            self.assertIn("tests pass", sent)
            self.assertIn("FAILED test_report.py::test_totals", sent)
            self.assertNotIn("report exists", sent)  # only what failed

    def test_a_run_that_died_after_doing_the_work_is_not_retried(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say(""),
                                    followups={"mode": "act"})
            report = h.run("Write report.csv.", contract=CONTRACT, run_checks=Workspace(h.ran))
            self.assertEqual((report.status, report.contract_met, report.followup), ("died", True, None), fw)

    def test_fixes_stop_at_max_fixes_when_the_contract_keeps_failing(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    say("Tried."), say("Tried again."), followups={"mode": "act"})
            report = h.run("Write report.csv.", contract=CONTRACT, run_checks=Workspace(h.ran, fail_until="never"),
                           max_fixes=2)
            self.assertEqual((len(report.verdicts), report.contract_met), (3, False), fw)

    def test_a_generated_contract_drops_unsafe_checks(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    followups={"mode": "observe"})
            writer = sdkt.ScriptedModel([sdkt.say(GENERATED)], name="writer") if fw == "sdk" else lgt.scripted(lgt.say(GENERATED))
            ws = Workspace(h.ran)
            report = h.run("Write report.csv.", contract="auto", contract_model=writer, run_checks=ws)
            self.assertEqual((report.contract.source, len(report.contract.checks), len(report.contract.refused)),
                             ("generated", 1, 1), fw)
            self.assertEqual(ws.seen, ["test -f report.csv"], fw)
            self.assertEqual([e["refused"] for e in h.logs if e["type"] == "contract"], [1], fw)

    def test_a_contract_that_cant_be_written_falls_back_to_the_usual_decision(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    followups={"mode": "observe"})
            writer = sdkt.ScriptedModel([sdkt.say("I can't do that.")], name="writer") if fw == "sdk" else lgt.scripted(lgt.say("nope"))
            report = h.run("Write report.csv.", contract="auto", contract_model=writer, run_checks=Workspace(h.ran))
            self.assertIsNone(report.contract, fw)
            self.assertEqual(report.decision["rule"], "verify", fw)   # the local rule, as without a contract
            self.assertTrue(any(e["type"] == "contract_unavailable" for e in h.logs), fw)

    def test_after_first_runs_before_any_follow_up_touches_the_workspace(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."),
                                    call("run_command", {"command": "python3 fix_totals.py"}, "c2"), say("Fixed."),
                                    followups={"mode": "act"})
            seen = []

            async def grade(report):
                seen.append((report.status, list(h.ran)))
            h.run("Write report.csv.", contract=CONTRACT, run_checks=Workspace(h.ran, fail_until="python3 fix_totals.py"),
                  after_first=grade)
            self.assertEqual(seen, [("done", ["python3 make_report.py"])], fw)   # the fix hadn't run yet
            self.assertIn("python3 fix_totals.py", h.ran, fw)

    def test_a_contract_needs_an_executor(self):
        h, _ = self.harness("sdk", sdkt.say("Done."))
        with self.assertRaises(ValueError):
            h.run("Write report.csv.", contract=CONTRACT)


if __name__ == "__main__":
    unittest.main()
