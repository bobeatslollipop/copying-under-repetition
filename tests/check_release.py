"""Check the bundled paper data without GPU access or API calls."""

import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plots"))
csv.field_size_limit(sys.maxsize)


class ReleaseDataTest(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            if ".venv" not in path.parts:
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(), filename=str(path))

    def test_data_checksums(self):
        manifest = json.loads((ROOT / "DATA_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            with self.subTest(path=entry["path"]):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_source_checksums(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            with self.subTest(path=entry["destination"]):
                path = ROOT / entry["destination"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 entry["release_sha256"])

    def test_release_text_hygiene(self):
        patterns = {
            "Chinese text": r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]",
            "API credential": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}",
            "Google API credential": r"\bAIza[0-9A-Za-z_-]{35}\b",
            "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
            "GitHub credential": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}",
            "private key": r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY",
            "local user path": re.escape("/" + "Users/") + "|" + re.escape("/" + "home-nfs/"),
        }
        compiled = {name: re.compile(pattern) for name, pattern in patterns.items()}
        for path in ROOT.rglob("*"):
            if not path.is_file() or {".venv", "outputs", ".git", "__pycache__"} & set(path.parts):
                continue
            if path.suffix not in {".py", ".md", ".json", ".txt", ".csv"} and path.name != ".gitignore":
                continue
            text = path.read_text(encoding="utf-8")
            for name, pattern in compiled.items():
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    self.fail(f"{path.relative_to(ROOT)}:{line}: {name}")

    def test_third_party_notices(self):
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())
        license_text = (ROOT / "licenses/Apache-2.0.txt").read_text()
        self.assertIn("Version 2.0, January 2004", license_text)
        self.assertIn("END OF TERMS AND CONDITIONS", license_text)

    def test_project_license(self):
        license_text = (ROOT / "LICENSE").read_text()
        self.assertIn("MIT License", license_text)
        self.assertIn("Permission is hereby granted, free of charge", license_text)
        self.assertIn("The above copyright notice and this permission notice", license_text)

    def test_llm_grid_and_scores(self):
        variants = json.loads((ROOT / "configs/llm.json").read_text())
        expected_cells = {(condition, str(length)) for condition in
                          ("iid", "geometric_L=100.0", "geometric_L=inf")
                          for length in (30,100,300,1000,3000)}
        for variant, settings in variants.items():
            with self.subTest(variant=variant):
                with (ROOT / "data/llm" / f"{variant}.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 750)
                counts = Counter((r["condition"],r["string_length"]) for r in rows)
                self.assertEqual(set(counts), expected_cells)
                self.assertEqual(set(counts.values()), {50})
                keys = {(r["condition"],r["string_length"],r["trial_idx"]) for r in rows}
                self.assertEqual(len(keys), 750)
                for row in rows:
                    self.assertEqual(row["model"], settings["model"])
                    self.assertEqual(len(row["target"]), int(row["string_length"]))
                    self.assertEqual(float(row["str_acc"]), float(row["prediction"] == row["target"]))
                    self.assertIn("raw_output", row)
                    self.assertIn("error", row)

    def test_synthetic_grids(self):
        from baseline_seed_results import load_results as baselines
        from vo_seed_results import load_results as vo
        self.assertEqual(len(baselines()), 45)
        self.assertEqual(sum(map(len,baselines().values())),16875)
        self.assertEqual(len(vo()), 9)
        self.assertEqual(sum(map(len,vo().values())),11250)

    def test_presets(self):
        baseline = json.loads((ROOT / "configs/synthetic/baseline.json").read_text())
        vo = json.loads((ROOT / "configs/synthetic/vo.json").read_text())
        for config in (baseline, vo):
            self.assertEqual(config["train_batch_size"] * config["gradient_accumulation_steps"],32)
            self.assertEqual(config["train_mixture_L_upper"],300)
            self.assertTrue(config["separate_eoi_eos"])
            self.assertEqual(config["eval_batch_size"] * config["eval_num_batches"] * config["eval_repeats"],96)
        self.assertEqual(vo["prime_start"],5)
        tree = ast.parse((ROOT / "synthetic_exps/models/common/vo.py").read_text())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="first_n_primes")
        self.assertEqual(ast.literal_eval(function.args.defaults[0]),5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
