import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from llm.run import main as llm_main
from synthetic_exps.evaluate import main as evaluate_main
from synthetic_exps.train import MODELS, ROOT, resolve_args


class CommandConfigurationTest(unittest.TestCase):
    def test_all_model_presets_and_seed(self):
        for model in MODELS:
            with self.subTest(model=model):
                _, args = resolve_args(["--model",model,"--seed","43","--dry-run"])
                self.assertEqual(args.model,model)
                self.assertEqual(args.seed,43)
                self.assertEqual(args.train_batch_size * args.gradient_accumulation_steps,32)
                self.assertEqual(args.max_train_len,2000)

    def test_all_llm_variants_have_complete_grid(self):
        variants = json.loads((ROOT / "configs/llm.json").read_text())
        for variant in variants:
            with self.subTest(variant=variant), contextlib.redirect_stdout(io.StringIO()) as stream:
                llm_main(["--variant",variant,"--dry-run"])
                metadata = json.loads(stream.getvalue())
                self.assertEqual(metadata["lengths"],[30,100,300,1000,3000])
                self.assertEqual(metadata["n_trials"],50)
                self.assertEqual(metadata["max_output_tokens"],20000)

    def test_evaluation_grids_are_inclusive(self):
        for model, count, last in (("nope",375,2993),("VO_1_no_scalar",1250,10000)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "args.json").write_text(json.dumps({"model":model,"seed":42}))
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    evaluate_main(["--run-dir",str(path),"--output",str(path/"new.csv"),"--dry-run"])
                metadata = json.loads(stream.getvalue())
                self.assertEqual(metadata["length_count"],count)
                self.assertEqual(metadata["max_length"],last)
                self.assertEqual(metadata["examples_per_length"],96)
                self.assertFalse((path/"new.csv").exists())


if __name__ == "__main__":
    unittest.main()
