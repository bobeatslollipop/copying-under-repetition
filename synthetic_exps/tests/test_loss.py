import unittest

import torch
import torch.nn.functional as F

from synthetic_exps.utils.train_utils import ce_loss


class MaskedCrossEntropyTest(unittest.TestCase):
    def test_loss_is_average_over_masked_labels_only(self):
        to_token = {"*": 0}
        labels = torch.tensor([[1, 2, 1, 0]])
        mask = torch.tensor([[1, 0, 1, 0]])
        logits = torch.tensor(
            [
                [
                    [0.0, 2.0, -1.0],
                    [50.0, -50.0, 50.0],
                    [0.0, 1.0, 3.0],
                    [-100.0, 100.0, -100.0],
                ]
            ],
            dtype=torch.float32,
        )

        actual = ce_loss(labels, logits, mask, to_token)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=to_token["*"],
            reduction="none",
        )
        expected = torch.sum(per_token * mask.reshape(-1)) / torch.sum(mask)
        self.assertTrue(torch.allclose(actual, expected))

        changed_unmasked = logits.clone()
        changed_unmasked[:, 1, :] = torch.tensor([-1000.0, 1000.0, -1000.0])
        changed_unmasked[:, 3, :] = torch.tensor([0.0, 0.0, 1000.0])
        self.assertTrue(torch.allclose(actual, ce_loss(labels, changed_unmasked, mask, to_token)))


if __name__ == "__main__":
    unittest.main()
