"""Outcome signals: the plugin's test/outcomes.test.js cases, plus the tracker's closes."""
import json
import unittest

from xybernetex.core.outcomes import OutcomeTracker, as_label, classify_followup, verify_result

REQUEST = "Write a script that parses the sales CSV and prints the total revenue per region"


class Clock:
    """Manual time and timers, so tests control quiet closes."""

    def __init__(self):
        self.t, self.timers = 1000.0, []

    def now(self):
        return self.t

    def set_timer(self, fn, seconds):
        h = [self.t + seconds, fn]
        self.timers.append(h)
        return h

    def clear_timer(self, h):
        if h in self.timers:
            self.timers.remove(h)

    def advance(self, seconds):
        self.t += seconds
        for h in [h for h in self.timers if h[0] <= self.t]:
            self.timers.remove(h)
            h[1]()


def tracker(**kw):
    c = Clock()
    logged, sent = [], []
    t = OutcomeTracker(log=logged.append, send=sent.append, now=c.now, set_timer=c.set_timer, clear_timer=c.clear_timer, **kw)
    return t, c, logged, sent


VERIFY = {"action": "verify", "probability": 0.9, "rule": "verify", "policy": "v1-uniform-2026-09-29"}
RUN = {"success": True, "retriable": False, "toolCalls": 2, "tokens": 12000}
FOLLOWUP = {"success": True, "toolCalls": 2, "writes": 1, "failedCalls": 0, "tokens": 5000}


class ClassifyTest(unittest.TestCase):
    def test_the_users_next_message_is_classified_locally(self):
        for msg in ["that didn't work, the totals are off", "It still fails on the header row", "you forgot the tests",
                    "No, I meant per country", "nope", "this is wrong", "doesnt work", "try again"]:
            self.assertEqual(classify_followup(REQUEST, msg), "correction", msg)
        self.assertEqual(classify_followup(REQUEST, "write a script that parses the sales CSV and prints total revenue per region please"),
                         "repeat")
        self.assertEqual(classify_followup(REQUEST, "thanks, perfect"), "thanks")
        self.assertEqual(classify_followup(REQUEST, "great"), "thanks")
        for msg in ["Now add a chart of the monthly totals", "fix the error in parse.py", "what's wrong with my config?",
                    "No problem is too small: write docs for the parser module and add examples",
                    "Make the report look great for the board meeting with charts per region"]:
            self.assertEqual(classify_followup(REQUEST, msg), "new", msg)

    def test_verify_result_files_changed_means_the_first_answer_needed_fixing(self):
        self.assertIsNone(verify_result(None))
        self.assertEqual(verify_result({"success": True, "writes": 2}), "fixed")
        self.assertEqual(verify_result({"success": True, "writes": 0}), "confirmed")
        self.assertEqual(verify_result({"success": False, "writes": 1}), "failed")

    def test_labels_keep_only_label_characters(self):
        self.assertEqual(as_label("@cf/zai-org/glm-5.3-flash"), "@cf/zai-org/glm-5.3-flash")
        self.assertEqual(as_label("decide-failed: boom (x)"), "decide-failed:_boom_x_")
        self.assertIsNone(as_label(""))


class TrackerTest(unittest.TestCase):
    def test_an_episode_joins_the_decision_the_followup_and_the_users_next_message(self):
        t, c, logged, sent = tracker()
        t.note_user_turn("s1", "build the parser and its tests")
        t.open("s1", agent_id="worker", model="@cf/zai-org/glm-5.3-flash", mode="act", decision=VERIFY, applied="verify",
               run=RUN, followup=FOLLOWUP)
        c.advance(90)
        ep = t.note_user_turn("s1", "thanks, looks good")
        self.assertEqual((ep["user"], ep["gapSec"], ep["applied"], ep["verify"], ep["probability"]), ("thanks", 90, "verify", "fixed", 0.9))
        self.assertEqual(ep["run"], RUN)
        self.assertEqual(ep["followup"], FOLLOWUP)
        self.assertEqual(logged[0]["type"], "episode")
        self.assertEqual(logged[0]["agentId"], "worker")
        self.assertEqual(t.open_episodes(), 0)
        t.flush()
        wire = json.dumps(sent[0])
        self.assertNotIn("parser", wire)  # the user's words never leave; only the label
        self.assertNotIn("looks good", wire)
        self.assertNotIn("sessionKey", wire)

    def test_silence_closes_an_episode_as_none(self):
        t, c, logged, _ = tracker(quiet_s=1800)
        t.open("s1", agent_id=None, model="m", mode="act", decision=VERIFY, applied="verify", run=RUN, followup=FOLLOWUP)
        c.advance(1799)
        self.assertEqual(t.open_episodes(), 1)
        c.advance(1)
        self.assertEqual(t.open_episodes(), 0)
        self.assertEqual((logged[0]["user"], logged[0]["gapSec"]), ("none", None))

    def test_a_held_out_run_keeps_its_propensity_and_observe_mode_is_untreated_for_certain(self):
        t, _, logged, _ = tracker()
        held_out = {"action": "none", "probability": 0.1, "rule": "verify-held-out", "policy": "p"}
        t.open("a", agent_id=None, model="m", mode="act", decision=held_out, applied="none", run=RUN, followup=None)
        t.open("b", agent_id=None, model="m", mode="observe", decision=VERIFY, applied="none", run=RUN, followup=None)
        t.open("c", agent_id=None, model="m", mode="observe", decision=held_out, applied="none", run=RUN, followup=None)
        t.flush()
        self.assertEqual([(e["sessionKey"], e["probability"], e["user"]) for e in logged],
                         [("a", 0.1, "session_end"), ("b", 1, "session_end"), ("c", 1, "session_end")])
        self.assertTrue(all(e["followup"] is None and e["verify"] is None for e in logged))

    def test_a_newer_decision_supersedes_and_a_crowd_evicts(self):
        t, _, logged, _ = tracker(max_tracked=2)
        for key in ("s1", "s1", "s2", "s3"):
            t.open(key, agent_id=None, model="m", mode="act", decision=VERIFY, applied="verify", run=RUN, followup=FOLLOWUP)
        self.assertEqual([(e["sessionKey"], e["user"]) for e in logged], [("s1", "superseded"), ("s1", "evicted")])
        t.end_session("s2")
        self.assertEqual(logged[-1]["user"], "session_end")

    def test_a_failing_sender_never_raises(self):
        def boom(_):
            raise OSError("offline")
        c = Clock()
        t = OutcomeTracker(send=boom, now=c.now, set_timer=c.set_timer, clear_timer=c.clear_timer)
        t.open("s1", agent_id=None, model="m", mode="act", decision=VERIFY, applied="verify", run=RUN, followup=FOLLOWUP)
        t.flush()


if __name__ == "__main__":
    unittest.main()
