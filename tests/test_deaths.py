"""The death-detection cases from the OpenClaw plugin's test/interventions.test.js."""
import unittest

from xybernetex.core.deaths import empty_run_error, last_reply_error, retriable


class DeathsTest(unittest.TestCase):
    def test_retriable_errors(self):
        self.assertTrue(retriable("incomplete_turn: model stopped"))
        self.assertTrue(retriable("Request timed out before a response was generated"))
        self.assertTrue(retriable("format error in tool call"))
        self.assertFalse(retriable("Run aborted by user"))
        self.assertFalse(retriable("approval denied"))
        self.assertFalse(retriable("user cancelled"))
        self.assertFalse(retriable(None))

    def test_an_empty_final_message_is_a_death_the_runtime_reported_as_success(self):
        user = {"role": "user", "content": "migrate the db"}
        empty = {"role": "assistant", "content": [], "stopReason": "length"}
        self.assertEqual(empty_run_error([user, empty]), "empty response from the model (stopReason length)")
        self.assertTrue(retriable(empty_run_error([user, empty])))
        # Any text or tool call means the model did answer.
        self.assertIsNone(empty_run_error([user, {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}]))
        self.assertIsNone(empty_run_error([user, {"role": "assistant", "content": [{"type": "toolCall", "id": "t1"}]},
                                           {"role": "toolResult", "content": []}, empty]))
        self.assertIsNone(empty_run_error([user, {"role": "assistant", "content": "plain string answer"}]))
        # Only this run counts: an earlier turn's answer doesn't rescue an empty one.
        self.assertEqual(empty_run_error([user, {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}, user, empty]),
                         "empty response from the model (stopReason length)")
        # No transcript, no user turn in view, or nothing from the model: can't tell.
        for m in (None, [], [empty], [user]):
            self.assertIsNone(empty_run_error(m))

    def test_the_fallback_reply_for_a_cut_off_answer_is_a_death_even_after_real_work(self):
        user = {"role": "user", "content": "write migrate.sql"}
        work = [{"role": "assistant", "content": [{"type": "toolCall", "id": "t1"}]}, {"role": "toolResult", "content": []}]
        cut_off = {"role": "assistant", "content": [{"type": "text", "text": "Create `migrate.sql` with:\n```sql\nBEGIN;"}],
                   "stopReason": "length"}
        fallback = {"role": "assistant", "stopReason": "stop", "idempotencyKey": "86e6a1e5:settled-finalization-fallback",
                    "content": [{"type": "text", "text": "The tool run finished, but no final summary was produced. "
                                                          "I did not repeat any completed actions."}]}
        err = empty_run_error([user, *work, cut_off, fallback])
        self.assertEqual(err, "no final answer: OpenClaw substituted its fallback reply (stopReason length)")
        self.assertTrue(retriable(err))
        # The marker alone is enough (the wording may change), and so is the wording alone (the key may).
        self.assertIn("no final answer", empty_run_error([user, *work, {**fallback, "content": [{"type": "text", "text": "Something else."}]}]))
        self.assertIn("no final answer", empty_run_error([user, *work, {**fallback, "idempotencyKey": None}]))
        # A real final answer after the same work is not a death; an earlier turn's fallback doesn't taint this one.
        self.assertIsNone(empty_run_error([user, *work, {"role": "assistant", "content": [{"type": "text", "text": "Done: migrate.sql written."}]}]))
        self.assertIsNone(empty_run_error([user, fallback, user, {"role": "assistant", "content": "All set."}]))

    def test_an_aborted_runs_reason_comes_from_its_final_message(self):
        user = {"role": "user", "content": "migrate the db"}
        work = [{"role": "assistant", "content": [{"type": "toolCall", "id": "t"}]}, {"role": "toolResult", "content": []}]
        timed_out = {"role": "assistant", "content": [{"type": "text", "text": ""}], "stopReason": "aborted",
                     "errorMessage": "request timed out"}
        self.assertEqual(last_reply_error([user, *work, timed_out]), "request timed out")
        self.assertTrue(retriable(last_reply_error([user, *work, timed_out])))
        stopped = {**timed_out, "errorMessage": "request aborted"}
        self.assertFalse(retriable(last_reply_error([user, *work, stopped])))
        self.assertIsNone(last_reply_error([user, *work]))  # last is a tool result, then nothing from the model
        self.assertIsNone(last_reply_error([user, {"role": "assistant", "content": []}]))
        self.assertIsNone(last_reply_error(None))


if __name__ == "__main__":
    unittest.main()
