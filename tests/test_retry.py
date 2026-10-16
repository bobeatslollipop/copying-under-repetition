import contextlib
import io
import json
import unittest
from unittest.mock import patch

import requests

from llm.core import DeepSeekClient, GeminiClient, OpenAIResponsesClient, run_trial


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode("utf-8")
    result.encoding = "utf-8"
    return result


PROVIDERS = (
    (
        OpenAIResponsesClient,
        {"status": "completed", "output_text": "abc"},
        {"status": "incomplete", "output_text": "ab",
         "incomplete_details": {"reason": "max_output_tokens"}},
        "incomplete response: max_output_tokens",
    ),
    (
        DeepSeekClient,
        {"choices": [{"message": {"content": "abc"}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "ab"}, "finish_reason": "length"}]},
        "DeepSeek finish_reason=length",
    ),
    (
        GeminiClient,
        {"candidates": [{"content": {"parts": [{"text": "abc"}]}, "finishReason": "STOP"}]},
        {"candidates": [{"content": {"parts": [{"text": "ab"}]}, "finishReason": "MAX_TOKENS"}]},
        "Gemini finishReason=MAX_TOKENS",
    ),
)


class APIRetryTest(unittest.TestCase):
    def run_mock_trial(self, client_type, responses):
        client = client_type(api_key="test-key", max_retries=1)
        with patch("llm.core.requests.post", side_effect=responses) as post, \
                patch("llm.core.time.sleep") as sleep, \
                contextlib.redirect_stdout(io.StringIO()):
            result = run_trial(client, "test-model", "abc", "iid", -1, 0, 0)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1.0)
        return result

    def test_successful_retry_clears_previous_request_error(self):
        for client_type, completed, _, _ in PROVIDERS:
            for failure in (
                requests.Timeout("initial timeout"),
                requests.ConnectionError("initial connection error"),
                response({"error": "rate limited"}, status=429),
                response({"error": "temporarily unavailable"}, status=503),
            ):
                with self.subTest(client=client_type.__name__, failure=str(failure)):
                    result = self.run_mock_trial(client_type, [failure, response(completed)])
                    self.assertEqual(result.error, "")
                    self.assertEqual(result.raw_output, "abc")
                    self.assertEqual(result.prediction, "abc")
                    self.assertEqual(result.str_acc, 1)

    def test_successful_retry_retains_current_generation_error(self):
        for client_type, _, incomplete, expected_error in PROVIDERS:
            with self.subTest(client=client_type.__name__):
                result = self.run_mock_trial(
                    client_type, [requests.Timeout("initial timeout"), response(incomplete)])
                self.assertEqual(result.error, expected_error)
                self.assertEqual(result.raw_output, "ab")
                self.assertEqual(result.prediction, "")
                self.assertEqual(result.str_acc, 0)

    def test_exhausted_retries_keep_last_request_error(self):
        for client_type, _, _, _ in PROVIDERS:
            with self.subTest(client=client_type.__name__):
                result = self.run_mock_trial(client_type, [
                    requests.Timeout("initial timeout"),
                    requests.ConnectionError("final connection error"),
                ])
                self.assertEqual(result.error, "final connection error")
                self.assertEqual(result.raw_output, "")
                self.assertEqual(result.prediction, "")
                self.assertEqual(result.str_acc, 0)


if __name__ == "__main__":
    unittest.main()
