"""Deletes of the agent's own files (authz.owns_files and the gate's "own_files"
waiver). The JavaScript plugin's test/own_files.test.js holds the same cases."""
import unittest

from xybernetex.core.authz import AuthorizationTracker
from xybernetex.core.control import ToolGate

REQUEST = "Build a small CLI for the key-value store in kv.py, with tests."


def sh(command):
    return {"command": command}


class Session:
    def __init__(self):
        self.t = AuthorizationTracker()
        self.t.set_request("s", REQUEST)

    def ran(self, command, failed=False):
        self.t.record_completed("s", "run_command", sh(command), failed)
        return self

    def wrote(self, path):
        self.t.record_completed("s", "write_file", {"path": path, "content": "x"})
        return self

    def owns(self, command, workdir=None):
        return self.t.owns_files("s", "run_command", {**sh(command), **({"workdir": workdir} if workdir else {})})


class OwnsFilesTest(unittest.TestCase):
    def test_a_plain_delete_of_files_the_agent_wrote_qualifies(self):
        s = Session().wrote("check_kv.py").ran("echo ok > out.txt")
        for cmd in ["rm check_kv.py", "rm -f check_kv.py", "rm -fv ./check_kv.py", "unlink out.txt",
                    "rm check_kv.py out.txt", "rm check_kv.py && rm out.txt", "sudo rm check_kv.py"]:
            self.assertTrue(s.owns(cmd), cmd)
        self.assertEqual(s.t.label("s", "run_command", sh("rm check_kv.py")), "own_artifact")

    def test_anything_more_than_a_plain_file_delete_does_not(self):
        s = Session().wrote("check_kv.py").ran("mkdir scratch")
        for cmd in ["rm -r check_kv.py", "rm -rf check_kv.py", "rm --recursive check_kv.py", "rm -d check_kv.py",
                    "rm -rf scratch", "rmdir scratch", "rm scratch",                  # folders never qualify
                    "rm check_kv.py kv.py",                                           # kv.py is the user's
                    "rm check_kv.py && python3 x.py", "rm check_kv.py && mv a b",   # only read-only company
                    "cd sub && rm check_kv.py", "cd .. && rm check_kv.py", "cd /tmp && rm check_kv.py", "ls",
                    "rm check_kv.py 2>/dev/null", "rm check_kv.py | tee log", "rm $(cat list)",
                    "rm check_*.py", "rm ../check_kv.py", "rm ~/check_kv.py", "rm $F", "rm -- -x",
                    "shred check_kv.py", "truncate -s 0 check_kv.py", "git rm check_kv.py"]:
            self.assertFalse(s.owns(cmd), cmd)
        self.assertFalse(s.t.owns_files("s", "write_file", {"path": "check_kv.py"}))
        self.assertFalse(AuthorizationTracker().owns_files("s", "run_command", sh("rm check_kv.py")))

    def test_the_working_folder_cd_and_read_only_company_are_understood(self):
        s = Session().wrote("runs/x/solve.py").wrote("check_kv.py")
        for cmd, wd in [("rm solve.py", "runs/x"), ("del solve.py", "runs/x"), ("Remove-Item solve.py", "runs/x"),
                        ("Remove-Item -Path solve.py -Force", "runs/x"), ("del /f /q solve.py", "runs/x"),
                        ("cd runs/x && rm solve.py", None), ("cd runs && rm x/solve.py && ls -la", None),
                        ("rm check_kv.py && ls", None), ("rm check_kv.py; cat out.txt", None), ("rm ../../check_kv.py", "runs/x")]:
            self.assertEqual(s.owns(cmd, wd), not cmd.startswith("rm ../"), (cmd, wd))
        for cmd, wd in [("rm solve.py", None), ("rm solve.py", "runs"), ("rm solve.py", "/abs/runs/x"),
                        ("rm solve.py", "../runs/x"), ("Remove-Item -Recurse solve.py", "runs/x"), ("del /s solve.py", "runs/x")]:
            self.assertFalse(s.owns(cmd, wd), (cmd, wd))
        # Exact paths only: a file made at tmp/data.csv never covers data.csv.
        self.assertFalse(Session().wrote("tmp/data.csv").owns("rm data.csv"))
        # Files a redirect made inside a cd'd folder are tracked there.
        self.assertTrue(Session().ran("cd out && echo x > log.txt").owns("rm out/log.txt"))

    def test_tool_caches_may_go_unless_a_move_named_them(self):
        for cmd in ["rm -rf __pycache__", "rm -rf runs/x/__pycache__", "rm -f kv.pyc", "Remove-Item -Recurse -Force __pycache__"]:
            self.assertTrue(Session().owns(cmd), cmd)
        for move in ["mv data.csv build/__pycache__/sub/", "mv data __pycache__", "mv -t . ../*"]:
            self.assertFalse(Session().ran(move).owns("rm -rf build/__pycache__"), move)
        self.assertFalse(Session().ran("mv data.csv kv.pyc").owns("rm -f kv.pyc"))

    def test_appends_touches_and_idempotent_forms_are_not_creations(self):
        s = Session().ran("echo x >> notes.txt").ran("touch data.csv").ran("cat kv.py 2> err.log")
        for cmd in ["rm notes.txt", "rm data.csv"]:
            self.assertFalse(s.owns(cmd), cmd)

    def test_a_folder_the_agent_made_cannot_launder_a_users_file(self):
        s = Session().ran("mkdir scratch").ran("mv data.csv scratch/")
        self.assertFalse(s.owns("rm -rf scratch"))
        self.assertFalse(s.owns("rm scratch/data.csv"))

    def test_a_move_onto_an_own_file_takes_it_out_of_the_set(self):
        cases = [
            ("echo x > data.csv", "mv ../data.csv .", "rm data.csv"),         # same file name, moved into this folder
            ("echo x > check.py", "mv kv.py check.py", "rm check.py"),        # same path
            ("echo x > out/log.txt", "mv backup out", "rm out/log.txt"),      # a folder above it
            ("echo x > a.txt", "mv -t . ../*", "rm a.txt"),                   # sources we can't read
            ("echo x > a.txt", "ls ../in | xargs mv -t .", "rm a.txt"),
            ("echo x > a.txt", "git mv ../a.txt a.txt", "rm a.txt"),
            ("echo x > a.txt", "rsync --remove-source-files ../a.txt .", "rm a.txt"),
        ]
        for create, move, delete in cases:
            s = Session().ran(create)
            self.assertTrue(s.owns(delete), create)
            s.ran(move)
            self.assertFalse(s.owns(delete), move)
        # A failed move may have moved part of its sources: it counts too.
        s = Session().ran("echo x > data.csv").ran("mv ../data.csv .", failed=True)
        self.assertFalse(s.owns("rm data.csv"))
        # An unrelated move leaves the set alone; rewriting an own file keeps it own.
        s = Session().ran("echo x > out.txt").ran("mv build.log logs/").wrote("out.txt")
        self.assertTrue(s.owns("rm out.txt"))


class GateTest(unittest.TestCase):
    def gate(self, preset="recommended", rules=None, approvals=False):
        s = Session()
        logs = []
        gate = ToolGate(mode="enforce", preset=preset, rules=rules, log=logs.append, approvals=approvals,
                        authorize=lambda e, c: s.t.label("s", e["tool_name"], e["params"]),
                        owns_files=lambda e, c: s.t.owns_files("s", e["tool_name"], e["params"]))
        call = lambda cmd, i: gate({"tool_name": "run_command", "params": sh(cmd), "tool_call_id": i},  # noqa: E731
                                   {"session_key": "s", "agent_id": "worker"})
        return s, call, logs

    def test_a_headless_agent_can_delete_its_own_files_but_not_its_own_folders(self):
        s, call, logs = self.gate()
        s.wrote("check_kv.py").ran("mkdir scratch")
        self.assertIsNone(call("rm check_kv.py", "c1"))
        waived = [e for e in logs if e["type"] == "tool_gate_waived"]
        self.assertEqual((waived[0]["authorization"], waived[0]["waiver"]), ("own_artifact", "own_files"))
        self.assertTrue(call("rm -rf scratch", "c2")["block"])
        self.assertEqual(logs[-1]["authorization"], "own_artifact")
        self.assertNotIn("waiver", logs[-1])

    def test_the_strict_preset_waives_it_too_and_a_rule_without_own_files_does_not(self):
        s, call, _ = self.gate(preset="strict")
        s.wrote("check_kv.py")
        self.assertIsNone(call("rm check_kv.py", "c1"))
        rule = {"id": "hold-deletes", "agentId": "*", "toolName": "*", "riskAtLeast": "destructive",
                "action": "block", "unlessAuthorization": ["requested"]}
        s, call, _ = self.gate(preset=None, rules=[rule])
        s.wrote("check_kv.py")
        self.assertTrue(call("rm check_kv.py", "c1")["block"])

    def test_with_approvals_the_own_file_delete_runs_without_a_prompt(self):
        s, call, _ = self.gate(approvals=True)
        s.wrote("check_kv.py")
        self.assertIsNone(call("rm check_kv.py", "c1"))
        s.ran("mkdir scratch")
        self.assertIn("require_approval", call("rm -rf scratch", "c2"))

    def test_rules_may_list_own_files(self):
        ToolGate(rules=[{"id": "r", "agentId": "*", "toolName": "*", "riskAtLeast": "destructive", "action": "block",
                         "unlessAuthorization": ["own_files"]}])
        with self.assertRaises(ValueError):
            ToolGate(rules=[{"id": "r", "agentId": "*", "toolName": "*", "riskAtLeast": "destructive", "action": "block",
                             "unlessAuthorization": ["own_artifact"]}])


if __name__ == "__main__":
    unittest.main()
