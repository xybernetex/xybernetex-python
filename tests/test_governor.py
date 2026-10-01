"""The governor: budgets, repeated calls and fix rounds without progress stop a
run, deterministically - in the core with a fake clock, and end to end
through both adapters."""
import json
import unittest
import warnings

warnings.filterwarnings("ignore")
import test_contract_runs as ctr  # noqa: E402

from xybernetex.core.governor import STANDARD, Governor, Limits, limits_from

CONTRACT = {"checks": [{"name": "tests pass", "command": "python3 -m pytest -q"}]}  # fails in ctr.Workspace until "never"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class GovernorTest(unittest.TestCase):
    def gov(self, **limits):
        logs, clock = [], Clock()
        return Governor(Limits(**limits), logs.append, now=clock), logs, clock

    def test_the_tool_call_budget_stops_the_call_that_crosses_it_and_every_one_after(self):
        g, logs, _ = self.gov(max_tool_calls=2)
        g.start("s")
        self.assertIsNone(g.on_call("s", "c1", "run", {"command": "a"}))
        self.assertIsNone(g.on_call("s", "c2", "run", {"command": "b"}))
        stop = g.on_call("s", "c3", "run", {"command": "c"})
        self.assertIn("Stopped by Xybernetex", stop)
        self.assertIn("2 tool calls", stop)
        self.assertEqual(g.on_call("s", "c4", "run", {"command": "d"}), stop)
        self.assertEqual(g.stopped("s"), "tool-calls")
        self.assertEqual([e["reason"] for e in logs], ["tool-calls"])   # logged once
        self.assertNotIn('"a"', json.dumps(logs))                        # never call arguments

    def test_the_same_call_id_gets_the_same_answer_and_counts_once(self):
        g, _, _ = self.gov(max_tool_calls=1)
        g.start("s")
        self.assertIsNone(g.on_call("s", "c1", "run", {"command": "a"}))
        self.assertIsNone(g.on_call("s", "c1", "run", {"command": "a"}))   # asked twice (approval, then guardrail)
        self.assertIsNotNone(g.on_call("s", "c2", "run", {"command": "b"}))

    def test_tokens_count_finished_turns_plus_the_current_one(self):
        g, _, _ = self.gov(max_tokens=1000)
        g.start("s")
        self.assertIsNone(g.on_call("s", "c1", "run", {}, turn_tokens=600))
        g.turn_done("s", 700)
        self.assertIsNone(g.on_call("s", "c2", "run", {"x": 1}, turn_tokens=200))
        self.assertIn("1000 model tokens", g.on_call("s", "c3", "run", {"x": 2}, turn_tokens=400))

    def test_time_is_measured_from_the_start_of_the_run(self):
        g, _, clock = self.gov(max_seconds=60)
        g.start("s")
        clock.t += 59
        self.assertIsNone(g.on_call("s", "c1", "run", {}))
        clock.t += 2
        self.assertIn("60 seconds", g.on_call("s", "c2", "run", {"x": 1}))

    def test_only_consecutive_identical_calls_count_as_a_repeat(self):
        g, _, _ = self.gov(repeat_limit=3)
        g.start("s")
        same = {"command": "pytest -q"}
        for i, params in enumerate([same, same, {"command": "cat log"}, same, same]):
            self.assertIsNone(g.on_call("s", f"c{i}", "run_command", params), i)
        self.assertIn("3 times in a row", g.on_call("s", "c9", "run_command", same))
        g.start("s")                                            # a new run starts clean
        self.assertIsNone(g.on_call("s", "d1", "run_command", same))

    def test_fix_rounds_without_progress_stop_the_run(self):
        g, logs, _ = self.gov(no_progress_rounds=2)
        g.start("s")
        self.assertIsNone(g.on_round("s", "same"))
        self.assertIsNone(g.on_round("s", "improved"))         # progress resets the count
        self.assertIsNone(g.on_round("s", "rolled-back"))
        self.assertEqual(g.on_round("s", "same"), "no-progress")
        self.assertEqual(logs[-1]["reason"], "no-progress")

    def test_limits_from_presets_dicts_and_mistakes(self):
        self.assertIsNone(limits_from(None))
        self.assertEqual(limits_from("standard"), STANDARD)
        custom = limits_from({"repeat_limit": 3})
        self.assertEqual((custom.repeat_limit, custom.max_tool_calls), (3, STANDARD.max_tool_calls))
        with self.assertRaises(ValueError):
            limits_from({"max_calls": 3})
        with self.assertRaises(ValueError):
            limits_from("strict")


class GovernedRunTest(ctr.ContractRunTest):
    def test_repeated_identical_calls_stop_the_run_and_nothing_follows(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, *[call("run_command", {"command": "python3 -m pytest -q"}, f"c{i}") for i in range(5)],
                                    say("I couldn't get the tests to pass."),
                                    followups={"mode": "act"}, governor={"repeat_limit": 3})
            report = h.run("Make the tests pass.")
            self.assertEqual(h.ran, ["python3 -m pytest -q"] * 2, fw)
            self.assertEqual((report.governor, report.decision["rule"], report.followup), ("repeat", "governor-repeat", None), fw)
            outputs = h.tool_outputs(report) if fw == "lg" else h.tool_outputs(report.result)
            self.assertEqual(sum("Stopped by Xybernetex" in str(o) for o in outputs), 3, fw)
            ready = [e for e in h.logs if e["type"] == "tool_gate_ready"][0]
            self.assertEqual(ready["governor"]["repeat_limit"], 3, fw)

    def test_the_tool_call_budget_holds_across_the_run(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, *[call("run_command", {"command": f"step {i}"}, f"c{i}") for i in range(4)],
                                    say("Stopped early."), governor={"max_tool_calls": 2})
            report = h.run("Do the four steps.")
            self.assertEqual((h.ran, report.governor), (["step 0", "step 1"], "tool-calls"), fw)

    def test_fix_rounds_without_progress_end_the_fix_loop(self):
        for fw in ("sdk", "lg"):
            call, say = self.steps(fw)
            h, model = self.harness(fw, call("run_command", {"command": "make report"}, "c1"), say("Done."),
                                    *[say(f"Tried {i}.") for i in range(5)],
                                    followups={"mode": "act"}, governor={"no_progress_rounds": 2})
            report = h.run("Write the report.", contract=CONTRACT, run_checks=ctr.Workspace(h.ran, fail_until="never"), max_fixes=5)
            self.assertEqual((len(report.ratchet), report.governor, report.contract_met), (2, "no-progress", False), fw)

    def test_without_a_governor_nothing_is_stopped(self):
        call, say = self.steps("sdk")
        h, model = self.harness("sdk", *[call("run_command", {"command": "python3 -m pytest -q"}, f"c{i}") for i in range(5)],
                                say("Done."))
        report = h.run("Make the tests pass.")
        self.assertEqual((len(h.ran), report.governor), (5, None))

    # The inherited contract tests run in test_contract_runs; skip them here.
    test_a_met_contract_ends_the_run_with_no_follow_up = None
    test_a_failed_contract_gets_a_targeted_fix_and_a_recheck = None
    test_a_run_that_died_after_doing_the_work_is_not_retried = None
    test_fixes_stop_at_max_fixes_when_the_contract_keeps_failing = None
    test_a_generated_contract_drops_unsafe_checks = None
    test_a_contract_that_cant_be_written_falls_back_to_the_usual_decision = None
    test_after_first_runs_before_any_follow_up_touches_the_workspace = None
    test_a_contract_needs_an_executor = None


if __name__ == "__main__":
    unittest.main()
