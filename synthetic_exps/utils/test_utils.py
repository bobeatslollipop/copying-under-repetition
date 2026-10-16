import copy
import torch
import numpy as np
import wandb
from .data_utils import get_eval_dataset


def find(s, ch):
    """Find all indices of character ch in string s."""
    return [i for i, ltr in enumerate(s) if ltr == ch]


def extract_gt_string(input_str):
    """
    Extract the original string (excluding gibberish) from input format.
    
    For current format: gibberish$string;gibberish:string.
    Legacy gibberish$string.gibberish:string. is also supported.
    Returns: the 'string' part (after :)
    """
    if ':' in input_str:
        start_pos = input_str.find(':')
    else:
        start_pos = input_str.find('.')
    if start_pos == -1:
        raise ValueError(f"Input string {input_str} does not contain : or .")
    return input_str[start_pos + 1:]


def record_first_eval_string(tokenizer, y, pred, step, model_type="latest", ignore_period=False):
    """
    Record model errors for analysis and upload to wandb.
    Always upload the first sample from the batch, regardless of error rate.
    """
    str_acc, char_acc, input_str, pred_ans = get_score(tokenizer, y, pred, 0, args=None, ignore_period=ignore_period)
    
    return {
        "step": step,
        "model_type": model_type,
        "input_string": input_str,
        "predicted_string": pred_ans,
        "str_acc": str_acc,
        "char_acc": char_acc,
    }


def get_score(tokenizer, y, pred, i, args=None, ignore_period=False, max_train_len=None, truncate_to_train_len=False):
    """Calculate string and character accuracy for a single prediction.
    
    Args:
        tokenizer: Tokenizer to decode token IDs
        y: Ground truth token IDs [batch, seq]
        pred: Predicted token IDs [batch, seq]
        i: Index of the sample in the batch
        ignore_period: If True, ignore the final period in ground truth
        max_train_len: Maximum training length (used when truncate_to_train_len=True)
        truncate_to_train_len: If True and string is longer than max_train_len,
            only compare the first max_train_len characters (or max_train_len+1 if 
            period is included via ignore_period=False)
    """
    y_str = tokenizer.decode(y[i])
    pred_str = tokenizer.decode(pred[i])

    try:
        if ':' in y_str:
            start_index = y_str.index(':')
        else:
            start_index = y_str.index('.')
        period_index = y_str.rindex('.')
    except ValueError as e:
        raise ValueError(f"Input string {y_str} does not contain : or .") from e

    if len(y_str) != len(pred_str):
        # For now, keeping the raise as it indicates a fundamental issue
        raise ValueError(f"Length of y_str and pred_str are not the same: {len(y_str)} != {len(pred_str)}")
    
    
    if ignore_period:
        # Ignore the period at the end of the ground truth
        # Original: gt was [start_index + 1 : period_index + 1] (includes period)
        # New:      gt is  [start_index + 1 : period_index]     (excludes period)
        gt = y_str[start_index + 1: period_index]
        pred_ans = pred_str[start_index + 1: period_index] # Also truncate prediction to match length/semantics
    else:
        gt = y_str[start_index + 1: period_index + 1]
        pred_ans = pred_str[start_index + 1: period_index + 1]
    
    # Truncate to max_train_len if requested and string is longer
    # For strings <= max_train_len, no change
    # For strings > max_train_len, only compare the first max_train_len characters
    if truncate_to_train_len and max_train_len is not None and len(gt) > max_train_len:
        gt = gt[:max_train_len]
        pred_ans = pred_ans[:max_train_len]
    
    str_acc = int(gt == pred_ans)
    char_acc = sum(int(a == b) for a, b in zip(gt, pred_ans)) / len(gt) if len(gt) > 0 else 0.0

    return str_acc, char_acc, gt, pred_ans


def _find_special_positions(y, colon_token_id, period_token_id):
    """
    Find the positions of colon and period tokens for each sample in the batch.
    Returns colon_positions and period_positions tensors of shape [batch_size].
    """
    batch_size = y.shape[0]
    seq_len = y.shape[1]
    device = y.device
    
    # Find colon positions (first occurrence per sample)
    colon_mask = (y == colon_token_id)
    # Use argmax on the mask - returns first True position (or 0 if none)
    # We need to handle the case where there's no match
    colon_positions = colon_mask.int().argmax(dim=1)
    
    # Find period positions (LAST occurrence per sample, which is the EOS)
    period_mask = (y == period_token_id)
    # Use weighted indices to find the last occurrence
    period_positions = (torch.arange(seq_len, device=device) * period_mask).argmax(dim=1)
    
    return colon_positions, period_positions


def batch_get_scores(y, pred, start_token_id, period_token_id, ignore_period=False, 
                     max_train_len=None, truncate_to_train_len=False):
    """
    Calculate string and character accuracy for a batch of predictions using tensor operations.
    
    Args:
        y: Ground truth token IDs [batch, seq]
        pred: Predicted token IDs [batch, seq]
        start_token_id: Token ID for colon ':' or period '.' (separating input and answer)
        period_token_id: Token ID for period '.'
        ignore_period: If True, ignore the final period in ground truth
        max_train_len: Maximum training length (used when truncate_to_train_len=True)
        truncate_to_train_len: If True and string is longer than max_train_len,
            only compare the first max_train_len characters
    
    Returns:
        str_acc_total: Sum of string accuracies for the batch
        char_acc_total: Sum of character accuracies for the batch
        batch_size: Number of samples processed
    """
    batch_size = y.shape[0]
    seq_len = y.shape[1]
    device = y.device
    
    # Find colon and period positions
    colon_positions, period_positions = _find_special_positions(y, start_token_id, period_token_id)
    
    # Calculate the length of comparison region for each sample
    # Region is [colon_pos + 1, period_pos] if ignore_period, else [colon_pos + 1, period_pos + 1]
    if ignore_period:
        end_positions = period_positions  # exclusive
    else:
        end_positions = period_positions + 1  # inclusive of period
    
    start_positions = colon_positions + 1
    compare_lengths = end_positions - start_positions  # length of region to compare

    
    # Apply truncation if needed
    if truncate_to_train_len and max_train_len is not None:
        compare_lengths = torch.clamp(compare_lengths, max=max_train_len)
    
    # Create a mask for positions to compare
    # position_indices[i, j] = j
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Mask: True if position is in comparison range
    # start_positions <= position < start_positions + compare_lengths
    mask = (position_indices >= start_positions.unsqueeze(1)) & \
           (position_indices < (start_positions + compare_lengths).unsqueeze(1))
    
    # Compare predictions with ground truth
    correct = (y == pred) & mask  # [batch, seq]
    
    # Character accuracy: sum of correct / length
    char_correct_per_sample = correct.sum(dim=1).float()  # [batch]
    char_acc_per_sample = char_correct_per_sample / compare_lengths.float().clamp(min=1)  # avoid div by 0
    char_acc_total = char_acc_per_sample.sum().item()
    
    # String accuracy: all positions must be correct
    # Count expected matches per sample vs actual matches
    str_acc_per_sample = (char_correct_per_sample == compare_lengths.float()).float()  # [batch]
    str_acc_total = str_acc_per_sample.sum().item()
    
    return str_acc_total, char_acc_total, batch_size


def _get_model_logits(model, x, args):
    """Get model logits based on model type."""
    if args.model == "mamba":
        return model(x)[0]
    else:
        return model(x)['logits']


def _evaluate_batch(model, x, y, args, suppress_period=False, period_token_id=None):
    """Evaluate a single batch and return predictions."""
    autocast_dtype = torch.float32
    if hasattr(args, 'precision'):
        if args.precision == 'bf16':
            autocast_dtype = torch.bfloat16
        elif args.precision == 'fp16':
            autocast_dtype = torch.float16
        elif args.precision == 'fp8':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16

    device_type = x.device.type if x.device.type != 'cpu' else 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
        logits = _get_model_logits(model, x, args)
        
        if suppress_period and period_token_id is not None:
             # If top prediction is period, choose second best
            top_2_logits, top_2_indices = torch.topk(logits, k=2, dim=-1)
            # top_2_indices: [batch, seq, 2]
            top1 = top_2_indices[..., 0]
            top2 = top_2_indices[..., 1]
            
            mask = (top1 == period_token_id)
            return torch.where(mask, top2, top1)
        
        return torch.argmax(logits, dim=-1)


def _evaluate_on_dataset(model, dataset, args, tokenizer, num_batches, return_first_sample=False, ignore_period=False, suppress_period=False, period_token_id=None, colon_token_id=None, max_train_len=None, truncate_to_train_len=False):
    """
    Helper to run evaluation on a dataset for a fixed number of batches.
    Returns mean metrics and optionally the first sample data.
    
    Uses batch_get_scores() for efficient tensor-based score computation.
    """
    total_examples = 0
    str_acc_total = 0.0
    char_acc_total = 0.0
    first_sample_data = None
    
    eval_iter = iter(dataset)
    device = next(model.parameters()).device
    
    for batch_idx in range(num_batches):
        try:
            eval_batch = next(eval_iter)
        except StopIteration:
            break
            
        x_eval = eval_batch['input_ids'][:, :-1].to(device, non_blocking=True)
        y_eval = eval_batch['input_ids'][:, 1:].to(device, non_blocking=True)
        
        pred_eval = _evaluate_batch(model, x_eval, y_eval, args, suppress_period=suppress_period, period_token_id=period_token_id)
        
        # Capture first sample of first batch if requested (still needs string output)
        if return_first_sample and batch_idx == 0:
            str_acc_ex, char_acc_ex, gt_ex, pred_ex = get_score(
                tokenizer, y_eval, pred_eval, 0, args=args, ignore_period=ignore_period,
                max_train_len=max_train_len, truncate_to_train_len=truncate_to_train_len
            )
            first_sample_data = {
                "input_string": gt_ex,
                "pred": pred_ex,
                "char_acc": char_acc_ex,
                "str_acc": str_acc_ex
            }

        # Use batch computation for scores (much faster than loop)
        start_token_id = colon_token_id
        batch_str_acc, batch_char_acc, batch_size_eval = batch_get_scores(
            y_eval, pred_eval, start_token_id, period_token_id,
            ignore_period=ignore_period,
            max_train_len=max_train_len,
            truncate_to_train_len=truncate_to_train_len
        )
        str_acc_total += batch_str_acc
        char_acc_total += batch_char_acc
        total_examples += batch_size_eval
            
    if total_examples == 0:
        return 0.0, 0.0, first_sample_data

    mean_str_acc = str_acc_total / total_examples
    mean_char_acc = char_acc_total / total_examples
    
    return mean_str_acc, mean_char_acc, first_sample_data


def quick_evaluation_with_multiple_probs(args, model, tokenizer, TO_TOKEN, model_type="latest", num_probs=5, num_batches=16, cumulative_tables=None, current_step=None, skip_tables=False):
    """Quick evaluation at three specific lengths for training loop."""
    ignore_period = False 
    suppress_period = False
    period_token_id = TO_TOKEN.get('.')
    colon_token_id = TO_TOKEN.get(':')

    # We will test three specific lengths
    max_len = getattr(args, 'max_train_len', 800)
    lengths = [max_len // 2, max_len, max_len * 2]
    
    # Skip table creation during training
    log_eval_examples = False if skip_tables else getattr(args, 'log_eval_examples', False)

    results = {}
    
    for cur_length in lengths:
        # Create a temporary args object for this specific length
        temp_args = copy.deepcopy(args)
        
        # Use training p_switch bounds and set random_switch=True for eval (same generation as training)
        if hasattr(args, 'train_switch_upper') and hasattr(args, 'train_switch_lower'):
            temp_args.eval_switch_upper = args.train_switch_upper
            temp_args.eval_switch_lower = args.train_switch_lower
        else:
            temp_args.eval_switch_upper = None
            temp_args.eval_switch_lower = None
            
        temp_args.eval_markov_random = True
        
        # Create evaluation dataset exactly for cur_length
        eval_dataset = get_eval_dataset(temp_args, tokenizer, TO_TOKEN,
                                      target_min_len=cur_length,
                                      target_max_len=cur_length)
        
        # Use helper
        mean_str_acc, mean_char_acc, first_sample = _evaluate_on_dataset(
            model, eval_dataset, temp_args, tokenizer, num_batches, return_first_sample=log_eval_examples, 
            ignore_period=ignore_period, suppress_period=suppress_period, period_token_id=period_token_id,
            colon_token_id=colon_token_id
        )
        
        examples_table = None
        if log_eval_examples and first_sample:
            # Create or get existing cumulative table for this length
            prob_key = f"{model_type}_len_{cur_length}"
            if cumulative_tables is not None and prob_key in cumulative_tables:
                examples_table = cumulative_tables[prob_key]
            else:
                examples_table = wandb.Table(
                    columns=["step", "input_string", "predicted", "char_acc", "str_acc", "string_length"],
                    log_mode="MUTABLE"
                )
                if cumulative_tables is not None:
                    cumulative_tables[prob_key] = examples_table
            
            step_value = current_step if current_step is not None else 0
            examples_table.add_data(
                step_value,
                first_sample["input_string"], 
                first_sample["pred"], 
                first_sample["char_acc"], 
                first_sample["str_acc"],
                int(cur_length)
            )
        
        # Store results for this length natively as a string key
        results[f"len_{cur_length}"] = {
            'mean_str_acc': mean_str_acc,
            'mean_char_acc': mean_char_acc,
            'examples_table': examples_table
        }
    
    return results


def quick_evaluation_iid(args, model, tokenizer, TO_TOKEN, model_type="latest", num_batches=16, cumulative_tables=None, current_step=None, skip_tables=False):
    """Quick evaluation on IID data for training loop."""
    ignore_period = False 
    suppress_period = False
    period_token_id = TO_TOKEN.get('.')
    colon_token_id = TO_TOKEN.get(':')
    
    # Skip table creation during training
    log_eval_examples = False if skip_tables else getattr(args, 'log_eval_examples', False)

    # Create evaluation dataset (uses defaults in args for IID)
    eval_dataset = get_eval_dataset(args, tokenizer, TO_TOKEN,
                                  target_min_len=args.min_train_len,
                                  target_max_len=args.max_train_len)
    
    mean_str_acc, mean_char_acc, first_sample = _evaluate_on_dataset(
        model, eval_dataset, args, tokenizer, num_batches, return_first_sample=log_eval_examples, 
        ignore_period=ignore_period, suppress_period=suppress_period, period_token_id=period_token_id,
        colon_token_id=colon_token_id
    )
    
    examples_table = None
    if log_eval_examples and first_sample:
        prob_key = f"{model_type}_iid"
        if cumulative_tables is not None and prob_key in cumulative_tables:
            examples_table = cumulative_tables[prob_key]
        else:
            examples_table = wandb.Table(
                columns=["step", "input_string", "predicted", "char_acc", "str_acc"],
                log_mode="MUTABLE"
            )
            if cumulative_tables is not None:
                cumulative_tables[prob_key] = examples_table
        
        step_value = current_step if current_step is not None else 0
        examples_table.add_data(
            step_value,
            first_sample["input_string"], 
            first_sample["pred"], 
            first_sample["char_acc"], 
            first_sample["str_acc"]
        )
    
    results = {
        'iid': {
            'mean_str_acc': mean_str_acc,
            'mean_char_acc': mean_char_acc,
            'examples_table': examples_table
        }
    }
    return results


def evaluation_with_multiple_probs(args, model, tokenizer, TO_TOKEN, model_type="latest", ignore_period=False):
    """Evaluate model performance with multiple switch probabilities, for testing"""
    if not (hasattr(args, 'eval_switch_upper') and hasattr(args, 'eval_switch_lower') and 
            args.eval_switch_upper is not None and args.eval_switch_lower is not None):
        # Fallback to single evaluation if no bounds are set
        return evaluation(args, model, tokenizer, TO_TOKEN, model_type, ignore_period=ignore_period)
    
    # Generate evenly spaced probabilities
    num_probs = getattr(args, 'num_eval_probs', 5) or 5 # Ensure fallback if None
    probs = np.linspace(args.eval_switch_lower, args.eval_switch_upper, num_probs)
    results = {}
    
    period_token_id = TO_TOKEN.get('.')
    colon_token_id = TO_TOKEN.get(':')
    # suppress_period logic is tied to ignore_period flag in this request
    suppress_period = ignore_period
    
    for prob_idx, switch_prob in enumerate(probs):
        # Create a temporary args object with this specific switch probability
        temp_args = copy.deepcopy(args)
        temp_args.eval_switch_upper = switch_prob
        temp_args.eval_switch_lower = switch_prob
        
        # Create evaluation dataset with this probability
        lengths = np.arange(args.min_eval_len, args.max_eval_len, 4)
        str_acc_mean_list = []
        char_accuracy_list = []
        
        repeats = getattr(args, 'eval_repeats', 5)
        
        # Create a table to log examples for this probability
        examples_table = wandb.Table(
            columns=["input_string", "predicted", "char_acc", "str_acc", "string_length", "switch_prob"],
            log_mode="MUTABLE"
        )
        
        for cur_length in lengths:
            # Store per-repeat metrics
            repeat_mean_list = []
            repeat_char_mean_list = []
            
            for rr in range(repeats):
                long_dataset = get_eval_dataset(temp_args, tokenizer, TO_TOKEN,
                                              target_min_len=cur_length, 
                                              target_max_len=cur_length)
                
                # Use helper
                # Note: we only need first sample for the very first repeat/batch of this length
                need_sample = (rr == 0)
                mean_str, mean_char, first_sample = _evaluate_on_dataset(
                    model, long_dataset, temp_args, tokenizer, args.eval_num_batches, return_first_sample=need_sample,
                    ignore_period=ignore_period, suppress_period=suppress_period, period_token_id=period_token_id,
                    colon_token_id=colon_token_id
                )

                repeat_mean_list.append(mean_str)
                repeat_char_mean_list.append(mean_char)
                
                if need_sample and first_sample:
                    examples_table.add_data(
                        first_sample["input_string"], 
                        first_sample["pred"], 
                        first_sample["char_acc"], 
                        first_sample["str_acc"], 
                        int(cur_length), 
                        float(switch_prob)
                    )
            
            # Average metrics across repeats
            mean_str_acc = float(np.mean(repeat_mean_list))
            mean_char_acc = float(np.mean(repeat_char_mean_list))
            
            str_acc_mean_list.append(mean_str_acc)
            char_accuracy_list.append(mean_char_acc)
            
            # Log evaluation results to wandb with probability-specific naming
            wandb.log({
                f"eval_string_accuracy_prob_{prob_idx}_{model_type}": mean_str_acc,
                f"eval_char_accuracy_prob_{prob_idx}_{model_type}": mean_char_acc,
                "eval_string_length": cur_length,
                "eval_switch_prob": round(switch_prob, 2)
            })
            
        # Log the examples table for this probability
        wandb.log({f"eval_table_prob_{prob_idx}_{model_type}": examples_table})
        
        # Store results for this probability (use actual probability value as key)
        results[switch_prob] = {
            'str_acc_mean_list': str_acc_mean_list,
            'char_accuracy_list': char_accuracy_list
        }
    
    return results


def evaluation(args, model, tokenizer, TO_TOKEN, model_type="latest", ignore_period=False, truncate_to_train_len=False):
    """Evaluate model performance across different sequence lengths."""
    # Use custom step size if available, otherwise default to 4
    step_size = getattr(args, 'eval_step_size', 4)
    lengths = np.arange(args.min_eval_len, args.max_eval_len, step_size)
    str_acc_mean_list = []
    char_accuracy_list = []
    print("\n")

    repeats = getattr(args, 'eval_repeats', 5)
    
    # Get max_train_len for truncation
    max_train_len = getattr(args, 'max_train_len', None)

    # Create a table to log one example per evaluated length (first batch, first sample)
    examples_table = wandb.Table(
        columns=["input_string", "predicted", "char_acc", "str_acc", "string_length"],
        log_mode="MUTABLE"
    )

    period_token_id = TO_TOKEN.get('.')
    colon_token_id = TO_TOKEN.get(':')
    suppress_period = ignore_period

    for cur_length in lengths:
        # Store per-repeat metrics
        repeat_mean_list = []
        repeat_char_mean_list = []

        for rr in range(repeats):
            long_dataset = get_eval_dataset(args, tokenizer, TO_TOKEN,
                                          target_min_len=cur_length, 
                                          target_max_len=cur_length)
            
            # Use helper
            need_sample = (rr == 0)
            mean_str, mean_char, first_sample = _evaluate_on_dataset(
                model, long_dataset, args, tokenizer, args.eval_num_batches, return_first_sample=need_sample,
                ignore_period=ignore_period, suppress_period=suppress_period, period_token_id=period_token_id,
                colon_token_id=colon_token_id, max_train_len=max_train_len, truncate_to_train_len=truncate_to_train_len
            )

            repeat_mean_list.append(mean_str)
            repeat_char_mean_list.append(mean_char)

            if need_sample and first_sample:
                 examples_table.add_data(
                    first_sample["input_string"], 
                    first_sample["pred"], 
                    first_sample["char_acc"], 
                    first_sample["str_acc"], 
                    int(cur_length)
                )

        # Average metrics across repeats
        mean_str_acc = float(np.mean(repeat_mean_list))
        mean_char_acc = float(np.mean(repeat_char_mean_list))

        str_acc_mean_list.append(mean_str_acc)
        char_accuracy_list.append(mean_char_acc)

        # Log evaluation results to wandb with model_type prefix
        wandb.log({
            f"eval_string_accuracy_{model_type}": mean_str_acc,
            f"eval_char_accuracy_{model_type}": mean_char_acc,
            "eval_string_length": cur_length,
        })

        print(f"len {cur_length}: str_acc {mean_str_acc}; char_acc {mean_char_acc}")
    print("\n")

    # Log the final examples table for this model type
    wandb.log({f"final_eval_table_{model_type}": examples_table})
    return str_acc_mean_list, char_accuracy_list
