import unittest
from argparse import Namespace
from unittest.mock import patch

from synthetic_exps.utils.data_utils import SingleStringTrainDataset, get_tokenizer


class CopyMaskTest(unittest.TestCase):
    def test_copy_mask_covers_answer_through_final_period(self):
        args = Namespace(vocab_size=3)
        tokenizer, to_token, _ = get_tokenizer(args)
        dataset = SingleStringTrainDataset(
            tokenizer,
            to_token,
            vocab_size=3,
            min_length=3,
            max_length=3,
            batch_size=1,
            length_answer=0,
            separate_eoi_eos=True,
        )

        batch = dataset[0]
        example = batch["input"][0]
        mask = batch["mask"][0]
        input_ids = batch["input_ids"][0]

        colon_pos = example.index(":")
        dot_pos = example.rindex(".")
        self.assertEqual(example[colon_pos + 1 : dot_pos], example[1:example.index(";")])

        expected_mask = [0] * len(mask)
        expected_mask[colon_pos + 1 : dot_pos + 1] = [1] * (dot_pos - colon_pos)
        self.assertEqual(mask.tolist(), expected_mask)
        self.assertEqual(mask[colon_pos].item(), 0)
        self.assertTrue(all(v == 0 for v in mask[dot_pos + 1 :].tolist()))

        shifted_labels = input_ids[1:]
        shifted_mask = mask[1:].bool()
        masked_label_ids = shifted_labels[shifted_mask]
        masked_text = tokenizer.decode(masked_label_ids)
        self.assertEqual(masked_text, example[colon_pos + 1 : dot_pos + 1])
        self.assertTrue(masked_text.endswith("."))

    def test_geometric_mode_samples_integer_L_inclusive(self):
        args = Namespace(vocab_size=3)
        tokenizer, to_token, _ = get_tokenizer(args)
        dataset = SingleStringTrainDataset(
            tokenizer,
            to_token,
            vocab_size=3,
            min_length=3,
            max_length=3,
            batch_size=1,
            length_answer=0,
            train_data_gen="geometric",
            L_lower=0,
            L_upper=300,
            separate_eoi_eos=True,
        )

        with patch("synthetic_exps.utils.data_utils.np.random.randint", return_value=0) as randint:
            self.assertEqual(dataset.get_current_L(), 0)
            randint.assert_called_once_with(0, 301)

    def test_auto_mode_preserves_continuous_L_sampling(self):
        args = Namespace(vocab_size=3)
        tokenizer, to_token, _ = get_tokenizer(args)
        dataset = SingleStringTrainDataset(
            tokenizer,
            to_token,
            vocab_size=3,
            min_length=3,
            max_length=3,
            batch_size=1,
            length_answer=0,
            train_data_gen="auto",
            L_lower=0,
            L_upper=300,
            separate_eoi_eos=True,
        )

        with patch("synthetic_exps.utils.data_utils.np.random.uniform", return_value=0.5) as uniform:
            self.assertEqual(dataset.get_current_L(), 0.5)
            uniform.assert_called_once_with(0, 300)


if __name__ == "__main__":
    unittest.main()
