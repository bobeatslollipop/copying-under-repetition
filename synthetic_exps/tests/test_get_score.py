import unittest


def simple_string_score(gt_str, pred_str):
    gt_index = gt_str.index(":")
    pred_index = pred_str.index(":")
    gt_content = gt_str[gt_index + 1 :]
    pred_content = pred_str[pred_index + 1 :]
    str_acc = int(gt_content == pred_content)
    char_acc = sum(map(str.__eq__, gt_content, pred_content)) / len(gt_content)
    return str_acc, char_acc


class GetScoreRegressionTest(unittest.TestCase):
    def test_suffix_mismatch_gives_partial_character_accuracy(self):
        s1 = (
            "gib:bababbabaabbbbbbbbbabbbbbaaabbbbbbbabbbaaabbaabaaabbbaaaabaaaaa"
            "abbabbbbabbaaaaabbbbabbbaaaaababaaaaaaaaaaabaaaaaaaaabaababbaaaabb"
            "baaaabaaaabbabaabbaaaabbbbbbbbaababaabbbbbaababbbbaabbbbaabbbbbba"
            "aaaabaabaaaaaaaabbabaaaabaaaabbbbbbbaabaaaaaaaaaaaaaaabaabbbaaabab"
            "aaaaaabbbaaaaaaaabababbbbabbbaaaabbbabaa."
        )
        s2 = (
            "gib:aaabbbbbbbabbbaaabbaabaaabbbaaaabaaaaaabbabbbbabbaaaaabbbbabbba"
            "aaaababaaaaaaaaaaabaaaaaaaaabaababbaaaabbbaaaabaaaabbabaabbaaaabbb"
            "bbbbbaababaabbbbbaababbbbaabbbbaabbbbbbaaaaabaabaaaaaaaabbabaaaab"
            "aaaabbbbbbbaabaaaaaaaaaaaaaaabaabbbaaababaaaaaabbbaaaaaaabbababbbb"
            "abbbaaaabbbabaa.........................."
        )
        str_acc, char_acc = simple_string_score(s1, s2)
        self.assertEqual(str_acc, 0)
        self.assertGreater(char_acc, 0.4)
        self.assertLess(char_acc, 0.5)


if __name__ == "__main__":
    unittest.main()
