"""Contract v2: checks must quote the request, a judge rules on failed checks
before any fix, and the agent may dispute a check (the judge decides) - the
fixes for v1's failure in the hard3 benchmark, where model-written checks
failed correct work and the fix turns then broke it."""
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

from xybernetex.core.contracts import (Check, Contract, CheckResult, Verdict, cites, contract_prompt, failure_message,
                                       parse_disputes, parse_generated, parse_judgment, restrict, without)
from test_contract_runs import Workspace, last_user_text

REQUEST = "Write report.csv and make the tests pass."
WRITTEN = json.dumps({"checks": [
    {"name": "report exists", "basis": "Write report.csv", "command": "test -f report.csv"},
    {"name": "tests pass", "basis": "make the tests pass", "command": "python3 -m pytest -q"},
    {"name": "made up", "basis": "sorted by date", "command": "grep -q 2026 report.csv"},
]})


class UnitTest(unittest.TestCase):
    def test_a_check_must_quote_the_request(self):
        self.assertTrue(cites("make the `tests` pass.", REQUEST))
        self.assertTrue(cites('"WRITE report.csv"', REQUEST))
        self.assertFalse(cites("sorted by date", REQUEST))
        self.assertFalse(cites("ok", REQUEST))
        self.assertFalse(cites(None, REQUEST))
        c = parse_generated(WRITTEN, REQUEST)
        self.assertEqual([x.name for x in c.checks], ["report exists", "tests pass"])
        self.assertEqual(c.checks[1].basis, "make the tests pass")
        self.assertIn("basis isn't a quote", c.refused[0])
        self.assertEqual(len(parse_generated(WRITTEN).checks), 3)  # v1 parsing is unchanged
        self.assertIn('"basis"', contract_prompt(REQUEST, "v2"))
        self.assertNotIn('"basis"', contract_prompt(REQUEST))

    def test_rulings_disputes_and_an_emptied_contract(self):
        self.assertEqual(parse_judgment('{"verdict": "check", "why": "the task never says"}'), "check")
        self.assertEqual(parse_judgment('```json\n{"verdict":"WORK"}\n```'), "work")
        self.assertEqual(parse_judgment("I think it's fine"), "work")   # unreadable: the check stands
        a, b = Check("tests pass", "pytest"), Check("report exists", "test -f r")
        contract = Contract(checks=(a, b))
        found = parse_disputes("All done.\nDISPUTE: Tests pass: the request never mentions pytest\n", contract)
        self.assertEqual(found, [(a, "the request never mentions pytest")])
        self.assertEqual(parse_disputes("no disputes", contract), [])
        verdict = Verdict(contract, [CheckResult(a, False, 1), CheckResult(b, True, 0)])
        kept = restrict(verdict, without(contract, [a]))
        self.assertTrue(kept.passed)
        emptied = restrict(verdict, without(contract, [a, b]))
        self.assertTrue(emptied.passed)                          # every check overturned: met
        self.assertFalse(Verdict(Contract(checks=())).passed)   # but never vacuously
        self.assertEqual(without(contract, [a]).hash, Contract(checks=(b,)).hash)
        message = failure_message(Verdict(Contract(checks=(Check("tests pass", "pytest", basis="make the tests pass"),)),
                                          [CheckResult(Check("tests pass", "pytest", basis="make the tests pass"), False, 1)]),
                                  disputable=True)
        self.assertIn("DISPUTE: <check name>: <why>", message)
        self.assertIn('checks this part of the request: "make the tests pass"', message)


class V2RunTest(unittest.TestCase):
    def setup(self, fw, *agent_turns):
        call, say = (sdkt.call, sdkt.say) if fw == "sdk" else (lgt.call, lgt.say)
        turns = [call("run_command", {"command": "python3 make_report.py"}, "c1"), say("Done."), *[say(t) for t in agent_turns]]
        if fw == "sdk":
            h = sdkt.Harness(*[[t] for t in turns], followups={"mode": "act"})
        else:
            h = lgt.Harness(*turns, followups={"mode": "act"})
        return h

    def models(self, fw, *rulings):
        if fw == "sdk":
            return (sdkt.ScriptedModel([sdkt.say(WRITTEN)], name="writer"),
                    sdkt.ScriptedModel(*[[sdkt.say(json.dumps({"verdict": r}))] for r in rulings], name="judge"))
        return lgt.scripted(lgt.say(WRITTEN)), lgt.scripted(*[lgt.say(json.dumps({"verdict": r})) for r in rulings])

    def test_a_check_the_judge_overturns_never_reaches_the_agent(self):
        for fw in ("sdk", "lg"):
            h = self.setup(fw)
            writer, judge = self.models(fw, "check")
            report = h.run(REQUEST, contract="auto", contract_model=writer, judge_model=judge, contract_version="v2",
                           run_checks=Workspace(h.ran, fail_until="never"))
            self.assertEqual((report.decision["rule"], report.contract_met, report.followup), ("contract-met", True, None), fw)
            self.assertEqual([j["ruling"] for j in report.judgments], ["check"], fw)
            self.assertEqual((len(report.contract.checks), report.contract.overturned), (1, 1), fw)
            self.assertIn("contract_judged", [e["type"] for e in h.logs], fw)

    def test_an_upheld_failure_gets_a_fix_and_a_disputed_check_goes_to_the_judge(self):
        for fw in ("sdk", "lg"):
            h = self.setup(fw, "Left the code as is.\nDISPUTE: tests pass: the request has no tests to run")
            writer, judge = self.models(fw, "work", "check")
            report = h.run(REQUEST, contract="auto", contract_model=writer, judge_model=judge, contract_version="v2",
                           run_checks=Workspace(h.ran, fail_until="never"), max_fixes=2)
            self.assertEqual(report.decision["rule"], "contract-failed", fw)
            self.assertIn("DISPUTE: <check name>", last_user_text(fw, h, h.model), fw)
            self.assertEqual([(j["round"], j["ruling"], j["disputed"]) for j in report.judgments],
                             [(0, "work", False), (1, "check", True)], fw)
            self.assertTrue(report.contract_met, fw)
            self.assertEqual(len(report.verdicts), 2, fw)   # one fix round, then met

    def test_v1_and_developer_contracts_are_never_judged(self):
        for fw in ("sdk", "lg"):
            h = self.setup(fw, "Tried.")
            writer, judge = self.models(fw, "check")
            report = h.run(REQUEST, contract={"checks": [{"name": "tests pass", "command": "python3 -m pytest -q"}]},
                           judge_model=judge, contract_version="v2", run_checks=Workspace(h.ran, fail_until="never"))
            self.assertEqual((report.judgments, report.contract_met), ([], False), fw)


if __name__ == "__main__":
    unittest.main()
