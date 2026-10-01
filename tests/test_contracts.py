"""Contracts: parsing and validation, the read-only filter, generated replies,
running checks through an executor, and the follow-up for a failed contract."""
import json
import unittest

from xybernetex.core.contracts import (Contract, ContractError, contract_prompt, failure_message, parse_contract,
                                       parse_generated, run_checks)
from xybernetex.core.followups import MARKER, is_ours

CHECKS = {"checks": [
    {"name": "report exists", "command": "test -f report.csv"},
    {"name": "header row", "command": "head -1 report.csv", "expect": r"^month,region,net_usd"},
    {"name": "tests pass", "command": "python3 -m pytest -q", "timeout": 120},
]}


def fake(results):
    """An executor answering by command: {command: (code, output) | Exception}."""
    seen = []

    def run(command, timeout):
        seen.append((command, timeout))
        r = results.get(command, (0, ""))
        if isinstance(r, Exception):
            raise r
        return r
    run.seen = seen
    return run


class ParseTest(unittest.TestCase):
    def test_a_contract_parses_and_its_hash_is_stable(self):
        c = parse_contract(CHECKS)
        self.assertEqual([k.name for k in c.checks], ["report exists", "header row", "tests pass"])
        self.assertEqual(c.checks[2].timeout, 120)
        self.assertEqual(c.source, "developer")
        self.assertEqual(len(c.hash), 16)
        self.assertEqual(c.hash, parse_contract(json.loads(json.dumps(CHECKS))).hash)
        self.assertNotEqual(c.hash, parse_contract({"checks": CHECKS["checks"][:2]}).hash)
        self.assertEqual(parse_contract(CHECKS["checks"]).hash, c.hash)  # a bare list works too

    def test_checks_that_write_or_destroy_are_refused(self):
        bad = ["rm -rf build", "git reset --hard", 'bash -c "rm -rf data"', 'echo "$(rm -rf data)"',
               "python3 make.py > report.csv", "cat a | tee report.csv", "cp report.csv backup.csv", "sed -i s/a/b/ f",
               "git commit -am done", "pip install pytest", "curl -X POST https://example.com", "mkdir out"]
        c = parse_contract({"checks": [{"command": b} for b in bad] + [{"command": "test -f report.csv"}]})
        self.assertEqual([k.command for k in c.checks], ["test -f report.csv"])
        self.assertEqual(len(c.refused), len(bad))
        self.assertTrue(all("read-only" in r for r in c.refused))
        self.assertNotIn("report.csv", " ".join(c.refused))  # reasons never echo the command
        with self.assertRaises(ContractError):
            parse_contract({"checks": [{"command": "rm -rf x"}]})

    def test_read_only_checks_including_null_redirects_are_kept(self):
        ok = ["python3 -m pytest -q 2>&1", "grep -q ACME- config.json", "python3 kv.py get a 2>/dev/null",
              "git status --porcelain", "node check.js >/dev/null", "ls out | wc -l"]
        self.assertEqual(len(parse_contract({"checks": [{"command": o} for o in ok]}).checks), len(ok))

    def test_malformed_checks_are_dropped_with_a_reason(self):
        c = parse_contract({"checks": [{"name": "no command"}, "string", {"command": "  "}, {"command": "x" * 3000},
                                       {"command": "test -f a", "expect": "("}, {"command": "test -f a", "expect": 5},
                                       {"command": "test -f a", "timeout": 10_000}]})
        self.assertEqual(len(c.checks), 1)
        self.assertEqual(c.checks[0].timeout, 60)  # an out-of-range timeout falls back to the default
        self.assertEqual(len(c.refused), 6)
        with self.assertRaises(ContractError):
            parse_contract({"steps": []})
        many = parse_contract({"checks": [{"command": f"test -f f{i}"} for i in range(20)]})
        self.assertEqual(len(many.checks), 12)
        self.assertEqual(len(many.refused), 8)


class GeneratedTest(unittest.TestCase):
    def test_the_prompt_carries_the_request_and_the_rules(self):
        p = contract_prompt("Write report.csv with columns month,region,net_usd.")
        self.assertIn("Write report.csv with columns month,region,net_usd.", p)
        self.assertIn("Read-only", p)
        self.assertIn('{"checks": [', p)

    def test_replies_parse_through_fences_and_prose(self):
        body = json.dumps(CHECKS)
        for reply in [body, f"```json\n{body}\n```", f"Here are the checks:\n{body}\nGood luck."]:
            c = parse_generated(reply)
            self.assertEqual((c.source, len(c.checks)), ("generated", 3))
        for bad in ["", "no json here", "{not json}", None]:
            with self.assertRaises(ContractError):
                parse_generated(bad)


class RunTest(unittest.TestCase):
    def test_all_checks_pass_and_the_verdict_says_so(self):
        c = parse_contract(CHECKS)
        run = fake({"head -1 report.csv": (0, "month,region,net_usd\n")})
        v = run_checks(c, run)
        self.assertTrue(v.passed)
        self.assertEqual([cmd for cmd, _ in run.seen], [k.command for k in c.checks])
        self.assertEqual(run.seen[2][1], 120)
        self.assertEqual(v.summary(), {"contract": c.hash, "checks": 3, "passed": 3, "failedAt": []})

    def test_exit_codes_patterns_and_executor_errors_fail_checks(self):
        c = parse_contract(CHECKS)
        v = run_checks(c, fake({"test -f report.csv": (1, ""), "head -1 report.csv": (0, "month,amount\n"),
                                "python3 -m pytest -q": TimeoutError("check timed out")}))
        self.assertFalse(v.passed)
        self.assertEqual(v.summary()["failedAt"], [0, 1, 2])
        self.assertEqual(v.results[2].error, "TimeoutError: check timed out")
        self.assertNotIn("report", json.dumps(v.summary()))  # the log line carries no check text

    def test_the_failure_message_names_what_failed_and_is_ours(self):
        c = parse_contract(CHECKS)
        v = run_checks(c, fake({"head -1 report.csv": (0, "month,amount\n"), "python3 -m pytest -q": (1, "1 failed\nFAILED test_x")}))
        m = failure_message(v)
        self.assertTrue(m.startswith(MARKER) and is_ours(m))   # never mistaken for the user's request
        self.assertIn("header row", m)
        self.assertIn("doesn't match", m)
        self.assertIn("FAILED test_x", m)
        self.assertNotIn("report exists", m)                  # passing checks aren't mentioned
        huge = run_checks(parse_contract({"checks": [{"name": f"c{i}", "command": f"test -f f{i}"} for i in range(12)]}),
                          fake({f"test -f f{i}": (1, "x" * 2000) for i in range(12)}))
        self.assertLessEqual(len(failure_message(huge)), 6000)

    def test_an_empty_contract_never_passes(self):
        self.assertFalse(run_checks(Contract(checks=()), fake({})).passed)


if __name__ == "__main__":
    unittest.main()
