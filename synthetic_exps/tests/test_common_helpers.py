import math
import unittest

import torch

from synthetic_exps.models.common import (
    apply_prime_head_penalty,
    build_causal_bias,
    build_hard_alibi_masks,
    causal_mask,
    make_head_primes,
    merge_heads,
    split_qkv,
)


class CommonAttentionHelperTest(unittest.TestCase):
    def test_split_qkv_and_merge_heads_shapes_and_order(self):
        qkv = torch.arange(2 * 3 * 24, dtype=torch.float32).view(2, 3, 24)
        query, key, value = split_qkv(qkv, num_attention_heads=4, head_size=2)

        self.assertEqual(query.shape, (2, 4, 3, 2))
        self.assertTrue(torch.equal(query[:, 0], qkv.view(2, 3, 4, 6)[:, :, 0, 0:2]))
        self.assertTrue(torch.equal(key[:, 0], qkv.view(2, 3, 4, 6)[:, :, 0, 2:4]))
        self.assertTrue(torch.equal(value[:, 0], qkv.view(2, 3, 4, 6)[:, :, 0, 4:6]))

        merged = merge_heads(query, num_attention_heads=4, head_size=2)
        self.assertEqual(merged.shape, (2, 3, 8))

    def test_causal_mask_extends_buffer(self):
        class Dummy:
            def __init__(self):
                self.bias = build_causal_bias(2)

            def _init_bias(self, max_positions, device=None):
                self.bias = build_causal_bias(max_positions, device=device)

        dummy = Dummy()
        mask = causal_mask(dummy, key_length=4, query_length=2, device=torch.device("cpu"))
        expected = torch.tensor(
            [[[[True, True, True, False], [True, True, True, True]]]]
        )
        self.assertEqual(dummy.bias.shape[-1], 4)
        self.assertTrue(torch.equal(mask, expected))

    def test_hard_alibi_masks_encode_offsets(self):
        masks = build_hard_alibi_masks(num_masked_heads=3, max_positions=5)
        self.assertTrue(masks[0, 3, 3])
        self.assertFalse(masks[0, 3, 2])
        self.assertTrue(masks[1, 3, 2])
        self.assertFalse(masks[1, 3, 1])
        self.assertTrue(masks[2, 3, 1])
        self.assertFalse(masks[2, 3, 0])
        self.assertFalse(masks[2, 1, 3])

    def test_prime_head_assignment_is_stable(self):
        primes = make_head_primes(8, num_primes=3, heads_per_prime=2)
        self.assertEqual(primes.tolist(), [5, 5, 7, 7, 11, 11, 1, 1])

    def test_vo_four_prime_heads_use_requested_periods(self):
        primes = make_head_primes(4, num_primes=4, heads_per_prime=1)
        self.assertEqual(primes.tolist(), [5, 7, 11, 13])

    def test_prime_head_penalty_only_affects_prime_heads(self):
        attn_scores = torch.zeros(1, 3, 3, 3)
        hidden_states = torch.zeros(1, 3, 4)
        scalar_weight = torch.zeros(2, 4)
        scalar_bias = torch.zeros(2)

        penalized = apply_prime_head_penalty(
            attn_scores,
            hidden_states,
            scalar_weight,
            scalar_bias,
            num_prime_heads=2,
        )
        penalty = torch.tensor(
            [
                [0.0, -math.log(2.0), -2.0 * math.log(2.0)],
                [math.log(2.0), 0.0, -math.log(2.0)],
                [2.0 * math.log(2.0), math.log(2.0), 0.0],
            ]
        )
        expected_scores = -penalty
        self.assertTrue(torch.allclose(penalized[0, 0], expected_scores))
        self.assertTrue(torch.allclose(penalized[0, 1], expected_scores))
        self.assertTrue(torch.equal(penalized[0, 2], attn_scores[0, 2]))

    def test_prime_head_penalty_noops_for_cached_lengths(self):
        attn_scores = torch.ones(1, 2, 2, 4)
        hidden_states = torch.zeros(1, 2, 4)
        out = apply_prime_head_penalty(
            attn_scores,
            hidden_states,
            torch.zeros(1, 4),
            torch.zeros(1),
            num_prime_heads=1,
        )
        self.assertTrue(torch.equal(out, attn_scores))


if __name__ == "__main__":
    unittest.main()
