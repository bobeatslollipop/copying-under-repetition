import unittest
from unittest.mock import patch

from llm.core import extract_output, run_trial, rand_str_k_letter_geometric


class LLMScoringTest(unittest.TestCase):
    def test_extraction_uses_target_length(self):
        self.assertEqual(extract_output("a\naaa\naaaaa", "aaa"), "aaa")

    def test_thinking_and_prefix_removal(self):
        self.assertEqual(extract_output("<think>reasoning</think>\nAnswer: `abc`", "abc"), "abc")

    def test_error_preserves_partial_raw_reply(self):
        class Client:
            def chat(self, **kwargs):
                return "abc", "timeout", ""
        result = run_trial(Client(), "test", "abc", "iid", -1, 0, 0)
        self.assertEqual(result.prediction, "")
        self.assertEqual(result.raw_output, "abc")
        self.assertEqual(result.str_acc, 0)

    def test_all_identical_sampler(self):
        with patch("llm.core.np.random.choice", return_value="b"):
            result = rand_str_k_letter_geometric(300,26,float("inf"))
        self.assertEqual(result, "b"*300)


if __name__ == "__main__":
    unittest.main()
