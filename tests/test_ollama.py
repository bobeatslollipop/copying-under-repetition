import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from llm.core import ExperimentConfig, OllamaClient, run_experiment, run_trial, summarize


def stream_response(chunks):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.iter_lines.return_value = [json.dumps(chunk).encode() for chunk in chunks]
    return response


class OllamaResponseTest(unittest.TestCase):
    def trial(self, chunks, use_chat=True, client=None):
        client = client or OllamaClient()
        with patch("llm.core.requests.post", return_value=stream_response(chunks)):
            result = run_trial(client, "test", "abc", "iid", -1, 0, 0,
                               use_chat_api=use_chat)
        return result, client.last_response_metadata

    def test_completed_answer_keeps_terminal_metadata(self):
        result, metadata = self.trial([
            {"message": {"content": "abc"}, "done": False},
            {"model": "test", "created_at": "2026-10-15T00:00:00Z", "done": True,
             "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 3,
             "total_duration": 1000, "eval_duration": 500},
        ])
        self.assertEqual(result.prediction, "abc")
        self.assertEqual(result.str_acc, 1)
        self.assertEqual(metadata["http_status"], 200)
        self.assertTrue(metadata["done"])
        self.assertEqual(metadata["done_reason"], "stop")
        self.assertEqual(metadata["eval_count"], 3)
        self.assertEqual(metadata["total_duration"], 1000)

    def test_empty_output_is_a_copying_failure(self):
        result, metadata = self.trial([
            {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        ])
        self.assertEqual(result.raw_output, "")
        self.assertEqual(result.prediction, "")
        self.assertEqual(result.error, "empty output")
        self.assertEqual(result.str_acc, 0)
        self.assertEqual(metadata["done_reason"], "stop")

    def test_thinking_without_an_answer_is_an_empty_output(self):
        result, _ = self.trial([
            {"message": {"thinking": "analysis"}, "done": False},
            {"done": True, "done_reason": "stop"},
        ])
        self.assertEqual(result.raw_output, "<think>analysis</think>\\n")
        self.assertEqual(result.prediction, "")
        self.assertEqual(result.error, "empty output")
        self.assertEqual(result.str_acc, 0)

    def test_stream_error_preserves_partial_text(self):
        for use_chat in (True, False):
            with self.subTest(use_chat=use_chat):
                content = {"message": {"content": "ab"}} if use_chat else {"response": "ab"}
                result, metadata = self.trial([
                    content, {"error": "generation failed"},
                ], use_chat=use_chat)
                self.assertEqual(result.error, "Ollama stream error: generation failed")
                self.assertEqual(result.raw_output, "ab")
                self.assertEqual(result.prediction, "")
                self.assertEqual(result.str_acc, 0)
                self.assertEqual(metadata["stream_error"], "generation failed")

    def test_error_only_stream_is_recorded(self):
        result, _ = self.trial([{"error": "generation failed"}])
        self.assertEqual(result.error, "Ollama stream error: generation failed")
        self.assertEqual(result.raw_output, "")
        self.assertEqual(result.str_acc, 0)

    def test_missing_completion_marker_is_recorded(self):
        result, _ = self.trial([{"message": {"content": "abc"}, "done": False}])
        self.assertEqual(result.error, "Ollama stream ended without a completion marker")
        self.assertEqual(result.raw_output, "abc")
        self.assertEqual(result.str_acc, 0)

    def test_generation_limit_keeps_partial_text(self):
        result, metadata = self.trial([
            {"message": {"content": "ab"}, "done": True, "done_reason": "length"},
        ])
        self.assertEqual(result.error, "Ollama done_reason=length")
        self.assertEqual(result.raw_output, "ab")
        self.assertEqual(result.str_acc, 0)
        self.assertEqual(metadata["done_reason"], "length")

    def test_metadata_is_reset_between_requests(self):
        client = OllamaClient()
        self.trial([{"message": {"content": "abc"}, "done": True,
                     "done_reason": "stop", "eval_count": 3}], client=client)
        with patch("llm.core.requests.post", side_effect=requests.ConnectionError("connection failed")):
            result = run_trial(client, "test", "abc", "iid", -1, 0, 1)
        self.assertEqual(result.error, "connection failed")
        self.assertNotIn("done_reason", client.last_response_metadata)
        self.assertNotIn("eval_count", client.last_response_metadata)
        self.assertEqual(client.last_response_metadata["request_error_type"], "ConnectionError")

    def test_summary_keeps_empty_outputs_and_request_errors(self):
        success, _ = self.trial([{"message": {"content": "abc"}, "done": True,
                                 "done_reason": "stop"}])
        empty, _ = self.trial([{"done": True, "done_reason": "stop"}])
        error, _ = self.trial([{"error": "500 Server Error: Internal Server Error"}])
        summary = summarize([success, empty, error])["test"]["iid"][3]
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["str_acc"], 1 / 3)

    def test_response_log_appends_without_replaying_empty_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(models=["test"], lengths=[3], L_values=[0.0],
                                      n_trials=1, run_name="test", output_dir=directory,
                                      verbose=False)
            with patch.object(OllamaClient, "is_available", return_value=True), \
                    patch.object(OllamaClient, "list_models", return_value=["test"]), \
                    patch("llm.core.requests.post", side_effect=lambda *a, **k: stream_response([
                        {"done": True, "done_reason": "stop", "eval_count": 0},
                    ])) as post, contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment(config)
                csv_path = Path(directory) / "test.csv"
                log_path = Path(directory) / "test.responses.jsonl"
                csv_before, log_before = csv_path.read_bytes(), log_path.read_bytes()
                entry = json.loads(log_before)
                self.assertEqual(entry["error"], "empty output")
                self.assertEqual(entry["response"]["eval_count"], 0)
                self.assertEqual(entry["target_sha256"], hashlib.sha256(first[0].target.encode()).hexdigest())
                config.resume = True
                run_experiment(config)
                self.assertEqual(post.call_count, 1)
                self.assertEqual((csv_path.read_bytes(), log_path.read_bytes()), (csv_before, log_before))
                config.n_trials = 2
                run_experiment(config)
                self.assertEqual(post.call_count, 2)
                self.assertTrue(csv_path.read_bytes().startswith(csv_before))
                self.assertTrue(log_path.read_bytes().startswith(log_before))
                self.assertEqual(len(log_path.read_text().splitlines()), 2)

    def test_partial_response_log_rejects_resume_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "test.responses.jsonl"
            log_path.write_text('{"partial":')
            config = ExperimentConfig(models=["test"], lengths=[3], L_values=[0.0],
                                      n_trials=1, run_name="test", output_dir=directory,
                                      resume=True, verbose=False)
            with patch.object(OllamaClient, "is_available", return_value=True), \
                    patch.object(OllamaClient, "list_models", return_value=["test"]), \
                    patch("llm.core.requests.post") as post, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "response log"):
                run_experiment(config)
            post.assert_not_called()
            self.assertEqual(log_path.read_text(), '{"partial":')
            self.assertFalse((Path(directory) / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()
