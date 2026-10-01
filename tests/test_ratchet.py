"""The ratchet: judging fix rounds by which checks pass, the rollback message,
folder snapshots, and - end to end through both adapters - a regressing fix
undone and a good one kept."""
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import test_contract_runs as ctr  # noqa: E402
import test_langgraph as lgt  # noqa: E402
import test_openai_agents as sdkt  # noqa: E402

from xybernetex.core.contracts import parse_contract, run_checks
from xybernetex.core.followups import MARKER, is_ours
from xybernetex.core.ratchet import FolderSnapshots, judge, passing, rollback_message

CONTRACT = {"checks": [{"name": "header row", "command": "check header"},
                       {"name": "totals", "command": "check totals"}]}


def verdict(header, totals):
    state = {"header": header, "totals": totals}
    return run_checks(parse_contract(CONTRACT), lambda cmd, t: (0 if state[cmd.split()[1]] else 1, ""))


class JudgeTest(unittest.TestCase):
    def test_sets_not_counts(self):
        self.assertEqual(judge(None, verdict(True, False)), "improved")              # the baseline
        self.assertEqual(judge(verdict(True, False), verdict(True, True)), "improved")
        self.assertEqual(judge(verdict(True, False), verdict(True, False)), "same")
        self.assertEqual(judge(verdict(True, False), verdict(False, False)), "regressed")
        self.assertEqual(judge(verdict(True, False), verdict(False, True)), "regressed")  # a trade is a regression
        self.assertEqual(passing(verdict(False, True)), frozenset({1}))

    def test_the_rollback_message_says_what_broke_and_what_still_fails(self):
        m = rollback_message(verdict(True, False), verdict(False, True))
        self.assertTrue(m.startswith(MARKER) and is_ours(m))
        self.assertIn("undone", m)
        self.assertIn("header row", m)
        self.assertIn("still fail", m)
        self.assertIn("totals", m)
        done = rollback_message(verdict(True, True), verdict(False, True))
        self.assertNotIn("still fail", done)


class FolderSnapshotsTest(unittest.TestCase):
    def test_restore_brings_back_exactly_the_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("v1")
            (root / "report.csv").write_text("month,region\n")
            snaps = FolderSnapshots(root)
            token = snaps.snapshot()
            (root / "report.csv").write_text("broken")
            (root / "src" / "a.py").unlink()
            (root / "junk").mkdir()
            (root / "junk" / "x").write_text("x")
            snaps.restore(token)
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")),
                             ["report.csv", "src", "src/a.py"])
            self.assertEqual((root / "report.csv").read_text(), "month,region\n")
            self.assertEqual((root / "src" / "a.py").read_text(), "v1")
            snaps.discard(token)
            self.assertFalse(Path(token).exists())
            snaps.close()


class World:
    """A pretend workspace the agent's commands change, with snapshots.

    write report -> header ok, totals wrong; rewrite all -> totals ok but the header broken;
    fix totals -> totals ok."""
    EFFECTS = {"write report": {"header": True, "totals": False}, "rewrite all": {"header": False, "totals": True},
               "fix totals": {"totals": True}}

    def __init__(self, ran):
        self.ran, self.done, self.state, self.snaps = ran, 0, {"header": False, "totals": False}, []

    def sync(self):
        for command in self.ran[self.done:]:
            self.state.update(self.EFFECTS.get(command, {}))
        self.done = len(self.ran)

    def __call__(self, command, timeout):   # the check executor
        self.sync()
        return (0 if self.state[command.split()[1]] else 1), ""

    def snapshot(self):
        self.sync()
        self.snaps.append(dict(self.state))
        return len(self.snaps) - 1

    def restore(self, token):
        self.sync()
        self.state = dict(self.snaps[token])

    def discard(self, token):
        pass


class RatchetRunTest(ctr.ContractRunTest):
    def script(self, fw):
        call, say = self.steps(fw)
        return (call("run_command", {"command": "write report"}, "c1"), say("Done."),
                call("run_command", {"command": "rewrite all"}, "c2"), say("Fixed the totals."),
                call("run_command", {"command": "fix totals"}, "c3"), say("Fixed the totals properly."))

    def test_a_regressing_fix_is_undone_and_a_good_one_kept(self):
        for fw in ("sdk", "lg"):
            h, model = self.harness(fw, *self.script(fw), followups={"mode": "act"})
            world = World(h.ran)
            report = h.run("Write report.csv with a header and correct totals.", contract=CONTRACT, run_checks=world,
                           snapshots=world, max_fixes=2)
            self.assertEqual((report.ratchet, report.contract_met), (["rolled-back", "improved"], True), fw)
            self.assertEqual(world.state, {"header": True, "totals": True}, fw)
            sent = ctr.last_user_text(fw, h, model)            # what the second fix turn was told
            self.assertIn("undone", sent)
            self.assertIn("header row", sent)
            outcomes = [e["outcome"] for e in h.logs if e["type"] == "ratchet"]
            self.assertEqual(outcomes, ["rolled-back", "improved"], fw)

    def test_without_snapshots_a_regression_is_seen_but_stays(self):
        for fw in ("sdk", "lg"):
            h, model = self.harness(fw, *self.script(fw), followups={"mode": "act"})
            world = World(h.ran)
            report = h.run("Write report.csv with a header and correct totals.", contract=CONTRACT, run_checks=world,
                           max_fixes=2)
            self.assertEqual((report.ratchet, report.contract_met), (["regressed", "same"], False), fw)
            self.assertEqual(world.state, {"header": False, "totals": True}, fw)

    # The inherited contract tests run again here; skip them.
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
