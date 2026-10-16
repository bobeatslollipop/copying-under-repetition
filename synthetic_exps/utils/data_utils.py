import numpy as np
import torch
import string
import random


GEOMETRIC_MIXTURE_DEFAULTS = {
    "iid_prob": 0.10,
    "repeated_prob": 0.80,
    "identical_prob": 0.10,
    "L_lower": 0.0,
    "L_upper": 500.0,
}

DEFAULT_SEPARATE_EOI_EOS = True



def get_tokenizer(args):
    string_ascii_lowercase = string.ascii_lowercase[:args.vocab_size]
    letters = dict(zip(string_ascii_lowercase, range(args.vocab_size)))

    symbols = {'$': len(letters), '|': len(letters)+1, '.': len(letters)+2, '*': len(letters)+3, ':': len(letters)+4, ';': len(letters)+5}

    # Preserve the historical vocabulary, including unused digit/operator tokens.
    base = len(letters) + len(symbols)
    for d in range(10):
        symbols[str(d)] = base + d
    symbols['+'] = base + 10
    symbols['='] = base + 11

    TO_TOKEN = {**letters, **symbols}

    TO_CHAR = {v: k for k, v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR


class NumberTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):
        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR

        self.bos_token_id = TO_TOKEN['$']
        self.eos_token_id = TO_TOKEN['.']
        self.dollar_token_id = TO_TOKEN.get('$')
        self.colon_token_id = TO_TOKEN.get(':')
        self.semicolon_token_id = TO_TOKEN.get(';')

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded_chars = []
        for t in x:
            if t in self.TO_CHAR:
                decoded_chars.append(self.TO_CHAR[t])
            else:
                # Handle out-of-vocab tokens - this shouldn't happen in normal operation
                print(f"Warning: Out-of-vocab token {t} encountered during decode")
                decoded_chars.append('?')  # Use '?' for unknown tokens
        return ''.join(decoded_chars)

    def __len__(self):
        return len(self.TO_TOKEN)


def arr_to_str(x):
    return ''.join([str(n) for n in x])


def rand_str_k_letter_geometric(length, vocab_size, L):
    """
    Generate a random string using k-letter geometric random walk.
    
    Unifies IID and Markov-like generation over all vocab_size characters.
    Each run of the same character has geometric length, then a new character
    is chosen uniformly at random from all vocab_size characters (including
    possibly the same one, with probability 1/vocab_size).
    
    Args:
        length: Length of the string to generate
        vocab_size: Number of characters (k) from 'a','b',...
        L: Expected run-length parameter.
            L=0: IID uniform (each character independently random)
            L=inf: constant string (all same character)
            L>0: geometric runs. At each step, probability 1/(L+1) of
                 switching to a new uniformly random character.
                 Expected run length = L+1.
    
    Returns:
        A string of the given length.
    """
    if length == 0:
        return ""
    
    alphabet = list(string.ascii_lowercase[:vocab_size])
    
    # Special case: IID uniform
    if L <= 0:
        chars = np.random.choice(alphabet, size=length)
        return ''.join(chars)
    
    # Special case: constant string
    if L == float('inf'):
        char = np.random.choice(alphabet)
        return char * length
    
    # General case: geometric runs
    # At each step, probability p_switch = 1/(L+1) of ending current run
    p_switch = 1.0 / (L + 1.0)
    
    chars = []
    remaining = length
    while remaining > 0:
        char = np.random.choice(alphabet)
        # Run length ~ Geometric(p_switch), minimum 1
        run_len = min(np.random.geometric(p_switch), remaining)
        chars.append(char * run_len)
        remaining -= run_len
    
    return ''.join(chars)


def sample_geometric_mixture_L(iid_prob=0.10, repeated_prob=0.80,
                               identical_prob=0.10, L_lower=0.0,
                               L_upper=500.0, return_component=False):
    """Sample the explicit iid / repeated / identical mixture used for diagnostics."""
    probs = np.asarray([iid_prob, repeated_prob, identical_prob], dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError("geometric mixture probabilities must be non-negative")
    if not np.isclose(float(probs.sum()), 1.0):
        raise ValueError(
            "geometric mixture probabilities must sum to 1.0; "
            f"got {float(probs.sum()):.6f}"
        )
    if L_upper < L_lower:
        raise ValueError("geometric mixture L_upper must be >= L_lower")

    component = np.random.choice(("iid", "repeated", "identical"), p=probs)
    if component == "iid":
        L = 0.0
    elif component == "identical":
        L = float("inf")
    else:
        L = float(np.random.uniform(L_lower, L_upper))

    if return_component:
        return L, component
    return L


def rand_str_two_letter_random_walk(length, vocab_size, switch_prob, letters_pair=None):
    """
    Generate a random string using a two-letter random walk.
    
    - Only two letters are used. If vocab_size > 2 and letters_pair is None,
      randomly pick two distinct letters from the first vocab_size letters.
    - With probability `switch_prob`, switch to the other letter; otherwise stay.
    
    Args:
        length: Length of the string to generate
        vocab_size: Size of vocabulary (number of available letters from 'a')
        switch_prob: Probability of switching to the other letter at each step
        letters_pair: Optional tuple/list of two letters to use (ensures gibberish uses same pair)
    """
    if length == 0:
        return ""
    
    if not 0 <= switch_prob <= 1:
        raise ValueError("switch_prob must be between 0 and 1")
    
    string_ascii_lowercase = list(string.ascii_lowercase[:vocab_size])
    
    if letters_pair is None:
        if len(string_ascii_lowercase) >= 2:
            letters_pair = list(np.random.choice(string_ascii_lowercase, size=2, replace=False))
        else:
            # Fallback if vocab_size < 2: use the single available letter twice
            only = string_ascii_lowercase[0]
            letters_pair = [only, only]
    else:
        # Sanity check
        if len(letters_pair) != 2:
            raise ValueError("letters_pair must contain exactly two letters")
    
    # Initialize with one of the two letters
    current = np.random.choice(letters_pair)
    chars = [current]
    
    for _ in range(length - 1):
        if np.random.random() < switch_prob:
            current = letters_pair[1] if current == letters_pair[0] else letters_pair[0]
        # else stay
        chars.append(current)
    
    return ''.join(chars)





def sample_str(len1, length_answer, vocab_size=26, gibberish_len=0,
               expected_switches=None, switch_prob=None, L=None,
               separate_eoi_eos=DEFAULT_SEPARATE_EOI_EOS):
    """
    Sample a copy-task string using k-letter geometric walk, two-letter random walk, or IID uniform.
    
    Args:
        len1: Length of the string to generate
        length_answer: Length of answer for copy task
        vocab_size: Size of vocabulary
        gibberish_len: Maximum length of gibberish strings to insert. Set to 0 to disable.
        expected_switches: If not None, use two-letter random walk with p = expected_switches / length.
        switch_prob: If not None, use two-letter random walk with this fixed switch probability.
        L: If not None, use k-letter geometric walk with this run-length parameter.
            L=0 gives IID uniform, L=inf gives constant string.
        separate_eoi_eos: If True, use ';' as the source end marker and '.'
            only as the answer EOS. This is the default for new copy runs.
            If False, preserve the legacy '.' source and answer terminator
            format.
    """
    # Choose string generation method
    if (expected_switches is not None) or (switch_prob is not None):
        # Use two-letter random walk; choose a single letter pair per example to share with gibberish
        string_ascii_lowercase = list(string.ascii_lowercase[:vocab_size])
        if len(string_ascii_lowercase) >= 2:
            chosen_pair = list(np.random.choice(string_ascii_lowercase, size=2, replace=False))
        else:
            only = string_ascii_lowercase[0]
            chosen_pair = [only, only]
        def generate_string(length):
            if expected_switches is not None:
                eff_p = float(expected_switches) / max(length, 1)
                eff_p = max(0.0, min(1.0, eff_p))
                return rand_str_two_letter_random_walk(length, vocab_size, eff_p, chosen_pair)
            else:
                return rand_str_two_letter_random_walk(length, vocab_size, switch_prob, chosen_pair)
    elif L is not None:
        # Use k-letter geometric random walk
        def generate_string(length):
            return rand_str_k_letter_geometric(length, vocab_size, L)
    else:
        # Default: IID uniform
        def generate_string(length):
            return rand_str_k_letter_geometric(length, vocab_size, 0)
    
    # Define start copy symbol
    start_copy = ":"
    source_end = ";" if separate_eoi_eos else "."

    num1 = generate_string(len1)

    # Generate gibberish strings if enabled
    if gibberish_len > 0:
        gibberish_len1 = np.random.randint(1, gibberish_len + 1)
        gibberish_len2 = np.random.randint(1, gibberish_len + 1)

        # Generate a gibberish base string long enough to provide both substrings.
        large_gibberish_length = max(len1, 2 * gibberish_len)
        large_gibberish = generate_string(large_gibberish_length)

        # Randomly select two substrings of the required lengths
        start1 = np.random.randint(0, len(large_gibberish) - gibberish_len1 + 1)
        gibberish1 = large_gibberish[start1:start1 + gibberish_len1]

        start2 = np.random.randint(0, len(large_gibberish) - gibberish_len2 + 1)
        gibberish2 = large_gibberish[start2:start2 + gibberish_len2]
    else:
        gibberish1 = ""
        gibberish2 = ""

    answer = num1[:length_answer] if length_answer > 0 else num1
    if gibberish_len > 0:
        example_str = f'{gibberish1}${num1}{source_end}{gibberish2}{start_copy}{answer}.'
    else:
        example_str = f'${num1}{source_end}{start_copy}{answer}.'

    return example_str


class SingleStringTrainDataset:
    """
    Training dataset for single-string mode. Each example contains exactly one task instance,
    padded with `*` up to a fixed maximum single-example length.
    """
    def __init__(self, tokenizer, TO_TOKEN, vocab_size=26, length_answer=0,
                 min_length=5, max_length=20, num_examples=1000,
                 batch_size=8, gibberish_len=0, expected_switches=None,
                 switch_upper=None, switch_lower=None,
                 L_upper=None, L_lower=None, train_data_gen="auto",
                 mixture_iid_prob=0.10, mixture_repeated_prob=0.80,
                 mixture_identical_prob=0.10, mixture_L_lower=0.0,
                 mixture_L_upper=500.0, separate_eoi_eos=DEFAULT_SEPARATE_EOI_EOS):
        self.tokenizer = tokenizer
        self.TO_TOKEN = TO_TOKEN
        self.vocab_size = vocab_size
        self.length_answer = length_answer
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.gibberish_len = gibberish_len
        self.expected_switches = expected_switches
        self.switch_upper = switch_upper
        self.switch_lower = switch_lower
        self.L_upper = L_upper
        self.L_lower = L_lower
        self.train_data_gen = train_data_gen
        self.mixture_iid_prob = mixture_iid_prob
        self.mixture_repeated_prob = mixture_repeated_prob
        self.mixture_identical_prob = mixture_identical_prob
        self.mixture_L_lower = mixture_L_lower
        self.mixture_L_upper = mixture_L_upper
        self.separate_eoi_eos = separate_eoi_eos


        # Compute the fixed padded length per example
        self.sequence_length = _compute_max_single_example_length(
            self.max_length, self.length_answer, self.gibberish_len
        )

        # Basic sanity: ensure min_length permits the task setup
        minimal_required_length = 0
        if self.length_answer and self.length_answer > 0:
            minimal_required_length += self.length_answer
        if self.min_length <= minimal_required_length:
            raise ValueError(
                f"Minimum length {self.min_length} must be greater than the required {minimal_required_length}."
            )


    def __len__(self):
        return self.num_examples
    
    def get_current_switch_prob(self):
        """Get the current two-letter random walk switch probability based on random selection from bounds"""
        if self.switch_upper is not None and self.switch_lower is not None:
            # Directly sample p uniformly from [lower, upper]
            return np.random.uniform(self.switch_lower, self.switch_upper)
        else:
            return None

    def get_current_L(self):
        """Sample the k-letter geometric run-length parameter for this example."""
        if self.train_data_gen == "geometric_mixture":
            return sample_geometric_mixture_L(
                iid_prob=self.mixture_iid_prob,
                repeated_prob=self.mixture_repeated_prob,
                identical_prob=self.mixture_identical_prob,
                L_lower=self.mixture_L_lower,
                L_upper=self.mixture_L_upper,
            )
        if self.train_data_gen == "geometric":
            if self.L_upper is None or self.L_lower is None:
                return None
            L_lower = int(self.L_lower)
            L_upper = int(self.L_upper)
            if L_upper < L_lower:
                raise ValueError("geometric L_upper must be >= L_lower")
            return int(np.random.randint(L_lower, L_upper + 1))
        if self.L_upper is not None and self.L_lower is not None:
            return np.random.uniform(self.L_lower, self.L_upper)
        return None

    def __getitem__(self, idx):
        batch = {
            'input': [],
            'input_ids': torch.full(
                (self.batch_size, self.sequence_length),
                self.TO_TOKEN['*'],
                dtype=torch.int64,
            ),
            'mask': torch.zeros((self.batch_size, self.sequence_length), dtype=torch.int64),
        }

        for batch_idx in range(self.batch_size):
            # Sample a single string
            len1 = self._sample_length()
            
            # Get current switch probability from curriculum scheduler if available
            current_switch_prob = self.get_current_switch_prob()
            # Sample L for k-letter geometric walk
            current_L = self.get_current_L()
            
            example_str = sample_str(
                len1,
                self.length_answer,
                self.vocab_size,
                self.gibberish_len,
                self.expected_switches,
                current_switch_prob,
                L=current_L,
                separate_eoi_eos=self.separate_eoi_eos,
            )

            # Tokenize and pad to fixed single-example length
            example_ids = self.tokenizer(example_str)
            len_ids = len(example_ids)
            batch['input_ids'][batch_idx, :len_ids] = example_ids

            # Build mask: 1s on the answer positions, 0 elsewhere (incl. padding)
            start_pos = example_str.find(':')
            
            dot_pos = example_str.rfind('.')
            if start_pos == -1 or dot_pos == -1:
                raise ValueError(f"Input string {example_str} does not contain suitable separator or .")
            # Mask 1s from start_pos+1 to dot
            batch['mask'][batch_idx, start_pos + 1:dot_pos + 1] = 1
            assert len_ids <= self.sequence_length
            batch['input'].append(example_str)

        return batch

    def _sample_length(self):
        if self.min_length == self.max_length:
            return self.min_length
            
        return int(np.random.randint(self.min_length, self.max_length + 1))


class EvalCopyDataset:
    def __init__(self, tokenizer, TO_TOKEN, vocab_size=26, length_answer=0,
                 sequence_length=220, min_length=8, max_length=30, 
                 num_examples=1000, batch_size=8, gibberish_len=0, expected_switches=None, switch_upper=None, switch_lower=None, random_switch=False, L=None, separate_eoi_eos=DEFAULT_SEPARATE_EOI_EOS):
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.TO_TOKEN = TO_TOKEN
        self.vocab_size = vocab_size
        self.length_answer = length_answer
        self.gibberish_len = gibberish_len
        self.expected_switches = expected_switches
        self.switch_upper = switch_upper
        self.switch_lower = switch_lower
        self.random_switch = random_switch
        self.L = L
        self.separate_eoi_eos = separate_eoi_eos

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {
            'input': [],
            'input_ids': torch.full(
                (self.batch_size, self.sequence_length),
                self.TO_TOKEN['*'],
                dtype=torch.int64,
            ),
            'mask': torch.zeros((self.batch_size, self.sequence_length), dtype=torch.int64),
        }
        minimal_required_length = _compute_max_single_example_length(
            self.max_length, self.length_answer, self.gibberish_len
        )
        if self.sequence_length < minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} because {minimal_required_length}>{self.sequence_length}. Increase your context length!")

        # For evaluation with upper/lower bounds, use 5 evenly spaced probabilities unless random_switch is True
        current_switch_prob = None
        if self.switch_upper is not None and self.switch_lower is not None:
            if not self.random_switch:
                # Generate 5 evenly spaced probabilities from lower to upper
                probs = np.linspace(self.switch_lower, self.switch_upper, 5)
                # Use the probability corresponding to the current batch index
                prob_idx = idx % 5
                current_switch_prob = probs[prob_idx]

        for batch_idx in range(self.batch_size):
            if self.switch_upper is not None and self.switch_lower is not None and self.random_switch:
                current_switch_prob = np.random.uniform(self.switch_lower, self.switch_upper)
            # Sample a string
            len1 = np.random.randint(self.min_length, self.max_length+1)
            example_str = sample_str(
                len1,
                self.length_answer,
                self.vocab_size,
                self.gibberish_len,
                self.expected_switches,
                current_switch_prob,
                L=self.L,
                separate_eoi_eos=self.separate_eoi_eos,
            )
            
            # Fill the context with padding
            example_ids = self.tokenizer(example_str)
            len_ids = len(example_ids)
            batch['input_ids'][batch_idx, :len_ids] = example_ids
            
            start_pos = example_str.find(':')
            
            dot_pos = example_str.rfind('.')
            if start_pos == -1 or dot_pos == -1:
                raise ValueError(f"Input string {example_str} does not contain suitable separator or .")

            # Mask 1s from start_pos+1 to dot (answer region)
            batch['mask'][batch_idx, start_pos + 1:dot_pos + 1] = 1
            assert len_ids <= self.sequence_length
            batch['input'].append(example_str)
        return batch


def _compute_max_single_example_length(max_len: int, length_answer: int, gibberish_len: int = 0) -> int:
    """
    Compute the maximum sequence length of a single copy-task example (including special tokens)
    for padding when using single-string mode.

    Format: {gibberish1}${num1}<EOI>{gibberish2}:{answer}.
    """
    la = int(length_answer) if length_answer is not None else 0
    
    if la <= 0:
        ans_length = 2 * max_len
    else:
        ans_length = max_len + la
        
    if gibberish_len > 0:
        return 2 * gibberish_len + ans_length + 4
    else:
        return ans_length + 4


def get_train_dataset(args, tokenizer):
    # We need TO_TOKEN to pad with '*'; rebuild from tokenizer
    # The tokenizer is NumberTokenizer, which exposes TO_TOKEN via attribute in this codebase
    TO_TOKEN = tokenizer.TO_TOKEN if hasattr(tokenizer, 'TO_TOKEN') else None
    if TO_TOKEN is None:
        raise ValueError("Tokenizer must expose TO_TOKEN for padding with '*'")

    train_dataset = SingleStringTrainDataset(
        tokenizer,
        TO_TOKEN,
        vocab_size=args.vocab_size,
        length_answer=args.length_answer,
        min_length=args.min_train_len,
        max_length=args.max_train_len,
        batch_size=args.train_batch_size,
        gibberish_len=getattr(args, 'train_gibberish_len', 0),
        expected_switches=getattr(args, 'train_expected_switches', None),
        switch_upper=getattr(args, 'train_switch_upper', None),
        switch_lower=getattr(args, 'train_switch_lower', None),
        L_upper=getattr(args, 'train_L_upper', None),
        L_lower=getattr(args, 'train_L_lower', None),
        train_data_gen=getattr(args, 'train_data_gen', 'auto'),
        mixture_iid_prob=getattr(args, 'train_mixture_iid_prob', GEOMETRIC_MIXTURE_DEFAULTS["iid_prob"]),
        mixture_repeated_prob=getattr(args, 'train_mixture_repeated_prob', GEOMETRIC_MIXTURE_DEFAULTS["repeated_prob"]),
        mixture_identical_prob=getattr(args, 'train_mixture_identical_prob', GEOMETRIC_MIXTURE_DEFAULTS["identical_prob"]),
        mixture_L_lower=getattr(args, 'train_mixture_L_lower', GEOMETRIC_MIXTURE_DEFAULTS["L_lower"]),
        mixture_L_upper=getattr(args, 'train_mixture_L_upper', GEOMETRIC_MIXTURE_DEFAULTS["L_upper"]),
        separate_eoi_eos=getattr(args, 'separate_eoi_eos', DEFAULT_SEPARATE_EOI_EOS),
        num_examples=args.steps,
    )
    return train_dataset


def get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len, target_max_len):
    gibberish_len = getattr(args, 'eval_gibberish_len', 0)
    eval_dataset = EvalCopyDataset(
        tokenizer,
        TO_TOKEN,
        vocab_size=args.vocab_size,
        length_answer=args.length_answer,
        sequence_length=_compute_max_single_example_length(target_max_len, args.length_answer, gibberish_len),
        min_length=target_min_len,
        max_length=target_max_len,
        batch_size=args.eval_batch_size,
        gibberish_len=gibberish_len,
        expected_switches=getattr(args, 'eval_expected_switches', None),
        switch_upper=getattr(args, 'eval_switch_upper', None),
        switch_lower=getattr(args, 'eval_switch_lower', None),
        random_switch=getattr(args, 'eval_markov_random', False),
        L=getattr(args, 'eval_L', None),
        separate_eoi_eos=getattr(args, 'separate_eoi_eos', DEFAULT_SEPARATE_EOI_EOS),
    )

    return eval_dataset
