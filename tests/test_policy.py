"""The policy-service client: the plugin's createRemoteDecider / createOutcomeSender cases."""
import json
import unittest

from xybernetex.core.followups import RunSummary
from xybernetex.core.policy import OutcomeSender, RemoteDecider, base_url


class Wire:
    """A fake post(): records requests, answers with a canned status and body (or raises)."""

    def __init__(self, status=200, body=None, error=None):
        self.status, self.body, self.error, self.sent = status, body or {}, error, []

    def __call__(self, url, api_key, body, timeout):
        self.sent.append({"url": url, "key": api_key, "body": body})
        if self.error:
            raise self.error
        return self.status, self.body


class RemoteDeciderTest(unittest.TestCase):
    def test_it_sends_only_the_runs_shape_and_returns_the_services_decision(self):
        wire = Wire(body={"action": "verify", "probability": 0.9, "rule": "verify", "policy": "v1-uniform-2026-09-29"})
        decide = RemoteDecider("k", "https://api.example/evaluate", post=wire)
        d = decide(RunSummary(success=True, tool_calls=4, model="m", error="secret error text"))
        self.assertEqual(d, {"action": "verify", "probability": 0.9, "rule": "verify", "policy": "v1-uniform-2026-09-29"})
        self.assertEqual(wire.sent[0]["url"], "https://api.example/intervene")
        self.assertEqual(wire.sent[0]["body"], {"summary": {"success": True, "retriable": False, "toolCalls": 4, "model": "m"}})
        self.assertNotIn("secret", json.dumps(wire.sent[0]["body"]))

    def test_a_death_is_sent_as_retriable_from_its_error(self):
        wire = Wire(body={"action": "retry", "probability": 0.9, "rule": "retry-on-death", "policy": "p"})
        RemoteDecider("k", post=wire)(RunSummary(success=False, tool_calls=1, error="empty response from the model"))
        self.assertEqual(wire.sent[0]["body"]["summary"]["retriable"], True)
        self.assertEqual(wire.sent[0]["url"], "https://api.xybernetex.com/intervene")

    def test_it_falls_back_to_the_local_rule_when_the_service_cant_answer(self):
        down = RemoteDecider("k", post=Wire(error=OSError("offline")),
                             fallback=lambda s: {"action": "retry", "probability": 0.9, "rule": "retry-on-death", "policy": "local"})
        f = down(RunSummary(success=False, tool_calls=1, retriable=True))
        self.assertEqual((f["action"], f["rule"], f["policy"]), ("retry", "fallback:retry-on-death", "local"))
        self.assertIn("offline", f["fallbackReason"])
        junk = RemoteDecider("k", post=Wire(body={"action": "delete-everything"}))
        self.assertTrue(junk(RunSummary(success=True, tool_calls=0))["rule"].startswith("fallback:"))
        refused = RemoteDecider("k", post=Wire(status=401, body={"error": "bad key"}))
        self.assertIn("HTTP 401 bad key", refused(RunSummary(success=True, tool_calls=3))["fallbackReason"])

    def test_base_url_accepts_the_root_or_the_plugin_style_evaluate_url(self):
        self.assertEqual(base_url("https://api.xybernetex.com/"), "https://api.xybernetex.com")
        self.assertEqual(base_url("https://api.xybernetex.com/evaluate"), "https://api.xybernetex.com")


class OutcomeSenderTest(unittest.TestCase):
    def test_it_posts_the_episode_and_raises_on_a_refusal(self):
        wire = Wire(status=204)
        OutcomeSender("k", "https://api.example", post=wire)({"action": "none"})
        self.assertEqual(wire.sent[0], {"url": "https://api.example/outcome", "key": "k", "body": {"episode": {"action": "none"}}})
        with self.assertRaises(RuntimeError):
            OutcomeSender("k", post=Wire(status=400, body={"error": "episode.user must be one of ..."}))({})


if __name__ == "__main__":
    unittest.main()
