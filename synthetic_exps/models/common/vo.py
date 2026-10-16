import torch


def first_n_primes(n, start=5):
    primes = []
    candidate = start
    while len(primes) < n:
        is_prime = True
        for k in range(2, int(candidate**0.5) + 1):
            if candidate % k == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def make_head_primes(num_attention_heads, num_primes=None, heads_per_prime=None):
    if num_primes is not None and heads_per_prime is not None:
        base_primes = first_n_primes(num_primes)
        primes_list = []
        for p in base_primes:
            primes_list.extend([p] * heads_per_prime)
    else:
        num_pairs = min(num_attention_heads // 2, 5)
        base_primes = first_n_primes(num_pairs)
        primes_list = []
        for p in base_primes:
            primes_list.extend([p, p])

    remaining = num_attention_heads - len(primes_list)
    if remaining > 0:
        primes_list.extend([1] * remaining)
    return torch.tensor(primes_list, dtype=torch.long)


def apply_prime_head_penalty(attn_scores, hidden_states, scalar_weight, scalar_bias, num_prime_heads):
    if hidden_states is None or num_prime_heads <= 0:
        return attn_scores

    query_length = attn_scores.size(-2)
    key_length = attn_scores.size(-1)
    if key_length != query_length:
        return attn_scores

    scalar_logits = torch.matmul(hidden_states, scalar_weight.t())
    scalar_logits = scalar_logits + scalar_bias.view(1, 1, -1)
    scalar_values = torch.nn.functional.softplus(scalar_logits).transpose(1, 2)
    scalar_values = scalar_values.to(attn_scores.dtype)
    cumsum_p = torch.cumsum(scalar_values, dim=-1)
    penalty = cumsum_p.unsqueeze(-1) - cumsum_p.unsqueeze(-2)

    prime_scores = attn_scores[:, :num_prime_heads] - penalty
    if num_prime_heads == attn_scores.size(1):
        return prime_scores
    return torch.cat((prime_scores, attn_scores[:, num_prime_heads:]), dim=1)
