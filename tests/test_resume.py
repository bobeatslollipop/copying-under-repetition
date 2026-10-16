import contextlib
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from llm.core import ExperimentConfig, TrialResult, load_results_from_csv, run_experiment
from llm.run import main


class FakeClient:
    def __init__(self):
        self.targets = []

    def is_available(self):
        return True

    def list_models(self):
        return ["gpt-5.5"]

    def chat(self, **kwargs):
        target = kwargs["user_message"].split("$", 1)[1].split(".", 1)[0]
        self.targets.append(target)
        return target, "", ""


def recorded_trial(index, error="", correct=True):
    target = "a" * 10
    prediction = target if correct and not error else ""
    return TrialResult("gpt-5.5", "iid", 10, -1.0, index, target, target,
                       prediction, float(bool(prediction)), float(bool(prediction)),
                       1.0 if prediction else 0.0, 0.0, error, 0.0)


def write_trials(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(TrialResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


class ResumeTest(unittest.TestCase):
    def run_config(self, directory, count, client, resume=True):
        config = ExperimentConfig(models=["gpt-5.5"], lengths=[10], L_values=[0.0],
                                  n_trials=count, provider="openai", openai_api_key="test",
                                  run_name="gpt-5.5", output_dir=directory,
                                  resume=resume, verbose=False)
        with patch("llm.core.OpenAIResponsesClient", return_value=client), contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(config)

    def test_resume_preserves_all_outcomes_and_appends_only_missing_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt-5.5.csv"
            rows = [recorded_trial(0), recorded_trial(1, "Gemini finishReason=MAX_TOKENS"),
                    recorded_trial(2, "DeepSeek finish_reason=length"),
                    recorded_trial(3, "Response ended prematurely"), recorded_trial(4, correct=False)]
            write_trials(path, rows)
            before = path.read_bytes()
            client = FakeClient()
            results = self.run_config(directory, 6, client)
            self.assertEqual(len(client.targets), 1)
            self.assertEqual(results[:5], rows)
            self.assertTrue(path.read_bytes().startswith(before))
            self.assertEqual(len(load_results_from_csv(str(path), strict=True)), 6)

    def test_complete_resume_does_not_rewrite_the_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt-5.5.csv"
            write_trials(path, [recorded_trial(0, "DeepSeek finish_reason=length")])
            before = path.read_bytes()
            client = FakeClient()
            self.run_config(directory, 1, client)
            self.assertEqual(client.targets, [])
            self.assertEqual(path.read_bytes(), before)

    def test_partial_invalid_and_duplicate_csv_fail_without_changes(self):
        for problem in ("partial", "invalid", "duplicate"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "gpt-5.5.csv"
                rows = [recorded_trial(0)] * (2 if problem == "duplicate" else 1)
                write_trials(path, rows)
                if problem == "partial":
                    path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
                elif problem == "invalid":
                    with path.open("a") as stream:
                        stream.write("invalid,row\n")
                before = path.read_bytes()
                client = FakeClient()
                with self.assertRaises(ValueError):
                    self.run_config(directory, 2, client)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(client.targets, [])

    def test_empty_interrupted_csv_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt-5.5.csv"
            path.touch()
            client = FakeClient()
            self.run_config(directory, 1, client)
            self.assertEqual(len(load_results_from_csv(str(path), strict=True)), 1)

    def test_fixed_seed_resumes_the_uninterrupted_target_sequence(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed:
            np.random.seed(123)
            client = FakeClient()
            full_results = self.run_config(full, 4, client, resume=False)
            write_trials(Path(resumed) / "gpt-5.5.csv", full_results[:2])
            np.random.seed(123)
            resumed_client = FakeClient()
            self.run_config(resumed, 4, resumed_client)
            self.assertEqual(resumed_client.targets, client.targets[2:])

    def test_changed_resume_settings_leave_metadata_and_csv_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("llm.core.run_experiment", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                main(["--variant", "gpt-5.5", "--output-dir", directory,
                      "--lengths", "30", "--n-trials", "1", "--seed", "123"])
            metadata = Path(directory) / "gpt-5.5.metadata.json"
            csv_path = Path(directory) / "gpt-5.5.csv"
            csv_path.write_text("recorded data\n")
            before = metadata.read_bytes(), csv_path.read_bytes()
            with patch("llm.core.run_experiment") as run, self.assertRaisesRegex(ValueError, "Existing settings differ"):
                main(["--variant", "gpt-5.5", "--output-dir", directory, "--resume",
                      "--lengths", "100", "--n-trials", "1", "--seed", "456"])
            run.assert_not_called()
            self.assertEqual((metadata.read_bytes(), csv_path.read_bytes()), before)

    def test_matching_resume_preserves_metadata_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["--variant", "gpt-5.5", "--output-dir", directory,
                         "--lengths", "30", "--n-trials", "1"]
            with patch("llm.core.run_experiment", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                main(arguments)
                metadata = Path(directory) / "gpt-5.5.metadata.json"
                metadata.write_text(json.dumps(json.loads(metadata.read_text())))
                before = metadata.read_bytes()
                main(arguments + ["--resume"])
            self.assertEqual(metadata.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
