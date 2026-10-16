from torch.optim import AdamW
import torch
import os
import time
import wandb
from pathlib import Path
from tqdm import tqdm
from .test_utils import quick_evaluation_with_multiple_probs, quick_evaluation_iid
import json
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import gc

class Float8CastSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if hasattr(torch, 'float8_e4m3fn'):
            try:
                orig_dtype = x.dtype
                return x.to(torch.float8_e4m3fn).to(orig_dtype)
            except Exception:
                pass
        
        # Bitwise truncation fallback
        if x.dtype == torch.float32:
            x_int = x.view(torch.int32)
            # 0xFFF00000 is -1048576 as signed 32-bit
            mask = torch.tensor(-1048576, dtype=torch.int32, device=x.device)
            return (x_int & mask).view(torch.float32)
        elif x.dtype == torch.bfloat16:
            x_int = x.view(torch.int16)
            # 0xFFF0 is -16 as signed 16-bit
            mask = torch.tensor(-16, dtype=torch.int16, device=x.device)
            return (x_int & mask).view(torch.bfloat16)
        elif x.dtype == torch.float16:
            # Retain one FP16 fraction bit while preserving its exponent range.
            x_int = x.view(torch.int16)
            # 0xFE00 is -512 as signed 16-bit
            mask = torch.tensor(-512, dtype=torch.int16, device=x.device)
            return (x_int & mask).view(torch.float16)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FP8Parametrization(torch.nn.Module):
    def forward(self, x):
        return Float8CastSTE.apply(x)

def apply_fp8_simulation(model):
    import torch.nn as nn
    import torch.nn.utils.parametrize as parametrize
    for module in model.modules():
        if isinstance(module, nn.Linear):
            def pre_hook(mod, inputs):
                return (Float8CastSTE.apply(inputs[0]),) + inputs[1:]
            module.register_forward_pre_hook(pre_hook)
            if not parametrize.is_parametrized(module, 'weight'):
                parametrize.register_parametrization(module, 'weight', FP8Parametrization())


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0


def log_memory_usage(step, prefix=""):
    """Log current memory usage with optional prefix."""
    memory_mb = get_memory_usage()
    if memory_mb > 0:
        print(f"{prefix}Step {step}: GPU Memory Usage: {memory_mb:.1f} MB")
    return memory_mb


def check_memory_threshold(threshold_mb=8000):
    """Check if memory usage exceeds threshold and suggest optimizations."""
    memory_mb = get_memory_usage()
    if memory_mb > threshold_mb:
        print(f"Warning: High GPU memory usage ({memory_mb:.1f} MB). Consider:")
        print("  - Using gradient checkpointing")
        print("  - Reducing batch size")
        print("  - Using smaller sequence lengths")
        return True
    return False


def _prepare_inputs_for_loss(inputs, logits, mask):
    """Helper to process inputs and logits for loss calculation."""
    # Handle different return types: dict, ModelOutput (with ['logits']), or tensor
    if isinstance(logits, dict):
        logits = logits['logits']
    elif hasattr(logits, '__getitem__') and not isinstance(logits, torch.Tensor):
        try:
            logits = logits['logits']
        except (KeyError, TypeError):
            pass

    # Flatten tensors
    shift_labels = inputs.reshape(-1)
    shift_logits = logits.reshape(-1, logits.size(-1))
    mask = mask.reshape(-1)
    
    return shift_labels, shift_logits, mask


def ce_loss(inputs, logits, mask, TO_TOKEN):
    shift_labels, shift_logits, mask = _prepare_inputs_for_loss(inputs, logits, mask)

    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=TO_TOKEN['*'],
        reduction='none',
    )
    return torch.sum(loss * mask) / torch.sum(mask)





class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.ema_params = {}
        self.original_params = {}
        
        # Initialize EMA parameters with current model parameters
        for name, param in model.named_parameters():
            name_clean = name.replace("module.", "")
            if param.requires_grad:
                self.ema_params[name_clean] = param.data.clone()
                self.original_params[name_clean] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                name_clean = name.replace("module.", "")
                if param.requires_grad and name_clean in self.ema_params:
                    self.ema_params[name_clean].mul_(self.beta).add_(param.data, alpha=1 - self.beta)
    
    def apply_ema(self, model):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            name_clean = name.replace("module.", "")
            if param.requires_grad and name_clean in self.ema_params:
                self.original_params[name_clean] = param.data.clone()
                param.data.copy_(self.ema_params[name_clean])
    
    def restore_original(self, model):
        """Restore original parameters to model"""
        for name, param in model.named_parameters():
            name_clean = name.replace("module.", "")
            if param.requires_grad and name_clean in self.original_params:
                param.data.copy_(self.original_params[name_clean])
    
    def save_ema_model(self, model, save_path):
        """Save model with EMA parameters"""
        # Apply EMA parameters
        self.apply_ema(model)
        
        # Save the model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(save_path)
        else:
            torch.save(model, save_path)
        
        # Restore original parameters
        self.restore_original(model)


def get_optimizer(model, args):
    weight_decay = getattr(args, 'weight_decay', 0.1)
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=weight_decay)
    return optimizer


class InverseSquareRootScheduler:
    """Custom learning rate scheduler with inverse square root decay"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, initial_lr=None):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr or optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.num_warmup_steps:
            # Warmup phase: linear increase from 0 to initial_lr
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps)
        else:
            # Decay phase: inverse square root decay
            # lr = initial_lr * sqrt(num_warmup_steps) / sqrt(step_count)
            lr = self.initial_lr * (self.num_warmup_steps ** 0.5) / (self.step_count ** 0.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'initial_lr': self.initial_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get('step_count', 0))
        self.num_warmup_steps = int(state_dict.get('num_warmup_steps', self.num_warmup_steps))
        self.num_training_steps = int(state_dict.get('num_training_steps', self.num_training_steps))
        self.initial_lr = float(state_dict.get('initial_lr', self.initial_lr))
        # Re-apply current lr based on step_count without incrementing
        if self.step_count <= self.num_warmup_steps:
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps if self.num_warmup_steps > 0 else 1.0)
        else:
            lr = self.initial_lr * (self.num_warmup_steps ** 0.5) / (self.step_count ** 0.5 if self.step_count > 0 else 1.0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class LinearDecayScheduler:
    """Learning rate scheduler with linear decay from initial_lr to 0"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, initial_lr=None):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr or optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.num_warmup_steps:
            # Warmup phase: linear increase from 0 to initial_lr
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps) if self.num_warmup_steps > 0 else self.initial_lr
        else:
            # Linear decay phase: from initial_lr to 0
            decay_steps = self.num_training_steps - self.num_warmup_steps
            steps_after_warmup = self.step_count - self.num_warmup_steps
            if decay_steps > 0:
                lr = self.initial_lr * (1 - steps_after_warmup / decay_steps)
                lr = max(lr, 0.0)  # Clamp to 0
            else:
                lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'initial_lr': self.initial_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get('step_count', 0))
        self.num_warmup_steps = int(state_dict.get('num_warmup_steps', self.num_warmup_steps))
        self.num_training_steps = int(state_dict.get('num_training_steps', self.num_training_steps))
        self.initial_lr = float(state_dict.get('initial_lr', self.initial_lr))
        # Re-apply current lr
        if self.step_count <= self.num_warmup_steps:
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps if self.num_warmup_steps > 0 else 1.0)
        else:
            decay_steps = self.num_training_steps - self.num_warmup_steps
            steps_after_warmup = self.step_count - self.num_warmup_steps
            if decay_steps > 0:
                lr = self.initial_lr * (1 - steps_after_warmup / decay_steps)
                lr = max(lr, 0.0)
            else:
                lr = self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class ConstantScheduler:
    """Constant learning rate scheduler (with optional warmup)"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, initial_lr=None):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr or optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.num_warmup_steps:
            # Warmup phase: linear increase from 0 to initial_lr
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps) if self.num_warmup_steps > 0 else self.initial_lr
        else:
            # Constant phase: stay at initial_lr
            lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'initial_lr': self.initial_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get('step_count', 0))
        self.num_warmup_steps = int(state_dict.get('num_warmup_steps', self.num_warmup_steps))
        self.num_training_steps = int(state_dict.get('num_training_steps', self.num_training_steps))
        self.initial_lr = float(state_dict.get('initial_lr', self.initial_lr))
        # Re-apply current lr
        if self.step_count <= self.num_warmup_steps:
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps if self.num_warmup_steps > 0 else 1.0)
        else:
            lr = self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineScheduler:
    """Cosine learning rate scheduler (with optional warmup)"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, initial_lr=None, min_lr=0.0):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr or optimizer.param_groups[0]['lr']
        self.min_lr = min_lr
        self.step_count = 0
        import math
        self.math = math
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.num_warmup_steps:
            # Warmup phase: linear increase from 0 to initial_lr
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps) if self.num_warmup_steps > 0 else self.initial_lr
        else:
            # Cosine decay phase
            decay_steps = self.num_training_steps - self.num_warmup_steps
            steps_after_warmup = self.step_count - self.num_warmup_steps
            if decay_steps > 0:
                # Clamp steps_after_warmup to decay_steps
                steps_after_warmup = min(steps_after_warmup, decay_steps)
                cosine = self.math.cos(self.math.pi * steps_after_warmup / decay_steps)
                lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cosine)
            else:
                lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get('step_count', 0))
        self.num_warmup_steps = int(state_dict.get('num_warmup_steps', self.num_warmup_steps))
        self.num_training_steps = int(state_dict.get('num_training_steps', self.num_training_steps))
        self.initial_lr = float(state_dict.get('initial_lr', self.initial_lr))
        self.min_lr = float(state_dict.get('min_lr', 0.0))
        
        # Re-apply current lr
        if self.step_count <= self.num_warmup_steps:
            lr = self.initial_lr * (self.step_count / self.num_warmup_steps if self.num_warmup_steps > 0 else 1.0)
        else:
            decay_steps = self.num_training_steps - self.num_warmup_steps
            steps_after_warmup = self.step_count - self.num_warmup_steps
            if decay_steps > 0:
                steps_after_warmup = min(steps_after_warmup, decay_steps)
                import math
                cosine = math.cos(math.pi * steps_after_warmup / decay_steps)
                lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cosine)
            else:
                lr = self.initial_lr
                
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class WSDScheduler:
    """Warmup-Stable-Decay (WSD) learning rate scheduler with plateau-triggered cooldown.

    Three phases:
        (a) Warmup:  linear 0 -> initial_lr over ``num_warmup_steps``.
        (b) Stable:  constant at ``initial_lr``. Duration is NOT predetermined;
                     it ends when ``start_cooldown()`` is called externally
                     (e.g. by a plateau detector) or when the safety net in
                     ``step()`` triggers.
        (c) Cooldown: linear decay from ``initial_lr`` to ``min_lr`` over
                      ``cooldown_length`` steps, where
                      ``cooldown_length = max(10_000, int(cooldown_fraction * step_at_trigger))``.
    """

    def __init__(self, optimizer, num_warmup_steps, max_training_steps,
                 initial_lr=None, min_lr=0.0, cooldown_fraction=0.2):
        self.optimizer = optimizer
        self.num_warmup_steps = int(num_warmup_steps)
        self.max_training_steps = int(max_training_steps)
        self.initial_lr = float(initial_lr if initial_lr is not None else optimizer.param_groups[0]['lr'])
        self.min_lr = float(min_lr)
        self.cooldown_fraction = float(cooldown_fraction)
        self.step_count = 0
        self.cooldown_start = None
        self.cooldown_length = None

    def start_cooldown(self):
        """Begin the cooldown phase. Idempotent: subsequent calls are no-ops."""
        if self.cooldown_start is None:
            self.cooldown_start = self.step_count
            self.cooldown_length = max(10_000, int(self.cooldown_fraction * self.step_count))

    def _compute_lr(self):
        # Warmup takes priority over any later phase.
        if self.step_count <= self.num_warmup_steps:
            if self.num_warmup_steps > 0:
                return self.initial_lr * (self.step_count / self.num_warmup_steps)
            return self.initial_lr
        # Cooldown phase
        if self.cooldown_start is not None:
            steps_in_cooldown = self.step_count - self.cooldown_start
            if self.cooldown_length and self.cooldown_length > 0:
                frac = min(steps_in_cooldown / self.cooldown_length, 1.0)
            else:
                frac = 1.0
            frac = max(frac, 0.0)
            return self.initial_lr + (self.min_lr - self.initial_lr) * frac
        # Stable phase
        return self.initial_lr

    def step(self):
        self.step_count += 1
        # Safety net: ensure cooldown completes within the step budget even
        # if plateau is never detected externally.
        if (self.cooldown_start is None
                and self.step_count + int(self.cooldown_fraction * self.step_count) >= self.max_training_steps):
            self.start_cooldown()

        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def is_done(self) -> bool:
        return (self.cooldown_start is not None
                and self.cooldown_length is not None
                and self.step_count >= self.cooldown_start + self.cooldown_length)

    def phase(self) -> int:
        """Return 0 for warmup, 1 for stable, 2 for cooldown."""
        if self.step_count <= self.num_warmup_steps:
            return 0
        if self.cooldown_start is not None:
            return 2
        return 1

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'num_warmup_steps': self.num_warmup_steps,
            'max_training_steps': self.max_training_steps,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'cooldown_fraction': self.cooldown_fraction,
            'cooldown_start': self.cooldown_start,
            'cooldown_length': self.cooldown_length,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict.get('step_count', 0))
        self.num_warmup_steps = int(state_dict.get('num_warmup_steps', self.num_warmup_steps))
        self.max_training_steps = int(state_dict.get('max_training_steps', self.max_training_steps))
        self.initial_lr = float(state_dict.get('initial_lr', self.initial_lr))
        self.min_lr = float(state_dict.get('min_lr', self.min_lr))
        self.cooldown_fraction = float(state_dict.get('cooldown_fraction', self.cooldown_fraction))
        cs = state_dict.get('cooldown_start', None)
        self.cooldown_start = int(cs) if cs is not None else None
        cl = state_dict.get('cooldown_length', None)
        self.cooldown_length = int(cl) if cl is not None else None
        # Re-apply current lr based on restored state
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class EmaImprovementPlateauDetector:
    """Detects plateau when train-loss EMA stops making significant progress."""

    detector_type = 'ema_improvement'

    def __init__(self, min_steps_before_check=800_000, patience_steps=200_000,
                 ema_beta=0.99, min_delta_log10=0.005, eps=1e-12):
        self.min_steps_before_check = int(min_steps_before_check)
        self.patience_steps = int(patience_steps)
        self.ema_beta = float(ema_beta)
        self.min_delta_log10 = float(min_delta_log10)
        self.eps = float(eps)
        self.ema_log_loss = None
        self.best_ema_log_loss = None
        self.best_step = None
        self.last_step = None
        self._last_improvement_log10 = float('nan')

    def update(self, step: int, loss: float) -> bool:
        import math
        step = int(step)
        loss_val = float(loss)
        log_loss = math.log10(max(loss_val, self.eps))
        self.last_step = step

        if self.ema_log_loss is None:
            self.ema_log_loss = log_loss
            self.best_ema_log_loss = log_loss
            self.best_step = step
            self._last_improvement_log10 = 0.0
            return False

        self.ema_log_loss = (
            self.ema_beta * self.ema_log_loss
            + (1.0 - self.ema_beta) * log_loss
        )
        improvement = self.best_ema_log_loss - self.ema_log_loss
        self._last_improvement_log10 = improvement
        if improvement >= self.min_delta_log10:
            self.best_ema_log_loss = self.ema_log_loss
            self.best_step = step
            self._last_improvement_log10 = 0.0

        if step < self.min_steps_before_check:
            return False
        if self.best_step is None:
            return False
        return (step - self.best_step) >= self.patience_steps

    def is_plateau(self) -> bool:
        if self.last_step is None or self.best_step is None:
            return False
        return (
            self.last_step >= self.min_steps_before_check
            and (self.last_step - self.best_step) >= self.patience_steps
        )

    def current_ema_log_loss(self):
        return float('nan') if self.ema_log_loss is None else self.ema_log_loss

    def best_log_loss(self):
        return float('nan') if self.best_ema_log_loss is None else self.best_ema_log_loss

    def steps_since_best(self):
        if self.last_step is None or self.best_step is None:
            return 0
        return int(self.last_step - self.best_step)

    def last_improvement_log10(self):
        return self._last_improvement_log10

    def state_dict(self):
        return {
            'detector_type': self.detector_type,
            'min_steps_before_check': self.min_steps_before_check,
            'patience_steps': self.patience_steps,
            'ema_beta': self.ema_beta,
            'min_delta_log10': self.min_delta_log10,
            'eps': self.eps,
            'ema_log_loss': self.ema_log_loss,
            'best_ema_log_loss': self.best_ema_log_loss,
            'best_step': self.best_step,
            'last_step': self.last_step,
            'last_improvement_log10': self._last_improvement_log10,
            'step_unit': 'micro_step',
        }

    def load_state_dict(self, state_dict):
        self.min_steps_before_check = int(state_dict.get('min_steps_before_check', self.min_steps_before_check))
        self.patience_steps = int(state_dict.get('patience_steps', self.patience_steps))
        self.ema_beta = float(state_dict.get('ema_beta', self.ema_beta))
        self.min_delta_log10 = float(state_dict.get('min_delta_log10', self.min_delta_log10))
        self.eps = float(state_dict.get('eps', self.eps))
        ema = state_dict.get('ema_log_loss', None)
        best = state_dict.get('best_ema_log_loss', None)
        best_step = state_dict.get('best_step', None)
        last_step = state_dict.get('last_step', None)
        self.ema_log_loss = float(ema) if ema is not None else None
        self.best_ema_log_loss = float(best) if best is not None else None
        self.best_step = int(best_step) if best_step is not None else None
        self.last_step = int(last_step) if last_step is not None else None
        self._last_improvement_log10 = float(state_dict.get('last_improvement_log10', float('nan')))


class EmaLossThresholdPlateauDetector:
    """Triggers cooldown once raw train-loss EMA drops below a fixed threshold."""

    detector_type = 'ema_loss_threshold'

    def __init__(self, loss_threshold=1e-4, ema_beta=0.99,
                 min_steps_before_check=0):
        self.loss_threshold = float(loss_threshold)
        self.ema_beta = float(ema_beta)
        self.min_steps_before_check = int(min_steps_before_check)
        self.ema_loss = None
        self.last_step = None

    def update(self, step: int, loss: float) -> bool:
        step = int(step)
        loss_val = float(loss)
        self.last_step = step
        if self.ema_loss is None:
            self.ema_loss = loss_val
        else:
            self.ema_loss = (
                self.ema_beta * self.ema_loss
                + (1.0 - self.ema_beta) * loss_val
            )
        return self.is_plateau()

    def is_plateau(self) -> bool:
        if self.last_step is None or self.ema_loss is None:
            return False
        if self.last_step < self.min_steps_before_check:
            return False
        return self.ema_loss < self.loss_threshold

    def current_ema_loss(self):
        return float('nan') if self.ema_loss is None else self.ema_loss

    def state_dict(self):
        return {
            'detector_type': self.detector_type,
            'loss_threshold': self.loss_threshold,
            'ema_beta': self.ema_beta,
            'min_steps_before_check': self.min_steps_before_check,
            'ema_loss': self.ema_loss,
            'last_step': self.last_step,
            'step_unit': 'micro_step',
        }

    def load_state_dict(self, state_dict):
        self.loss_threshold = float(state_dict.get('loss_threshold', self.loss_threshold))
        self.ema_beta = float(state_dict.get('ema_beta', self.ema_beta))
        self.min_steps_before_check = int(state_dict.get('min_steps_before_check', self.min_steps_before_check))
        ema = state_dict.get('ema_loss', None)
        last_step = state_dict.get('last_step', None)
        self.ema_loss = float(ema) if ema is not None else None
        self.last_step = int(last_step) if last_step is not None else None


def make_plateau_detector(args):
    detector_type = getattr(args, 'plateau_detector', 'ema_improvement')
    if detector_type == 'ema_loss_threshold':
        return EmaLossThresholdPlateauDetector(
            loss_threshold=getattr(args, 'plateau_loss_threshold', 1e-4),
            ema_beta=getattr(args, 'plateau_ema_beta', 0.99),
            min_steps_before_check=getattr(args, 'plateau_min_steps', 0),
        )
    if detector_type == 'ema_improvement':
        return EmaImprovementPlateauDetector(
            min_steps_before_check=getattr(args, 'plateau_min_steps', 800_000),
            patience_steps=getattr(args, 'plateau_patience_steps', 200_000),
            ema_beta=getattr(args, 'plateau_ema_beta', 0.99),
            min_delta_log10=getattr(args, 'plateau_min_delta_log10', 0.005),
        )
    raise ValueError(f"Unknown plateau detector: {detector_type}")


def refresh_plateau_detector_config(plateau_detector, args):
    if isinstance(plateau_detector, EmaImprovementPlateauDetector):
        plateau_detector.min_steps_before_check = int(getattr(args, 'plateau_min_steps', 800_000))
        plateau_detector.patience_steps = int(getattr(args, 'plateau_patience_steps', 200_000))
        plateau_detector.ema_beta = float(getattr(args, 'plateau_ema_beta', 0.99))
        plateau_detector.min_delta_log10 = float(getattr(args, 'plateau_min_delta_log10', 0.005))
    elif isinstance(plateau_detector, EmaLossThresholdPlateauDetector):
        plateau_detector.min_steps_before_check = int(getattr(args, 'plateau_min_steps', 0))
        plateau_detector.ema_beta = float(getattr(args, 'plateau_ema_beta', 0.99))
        plateau_detector.loss_threshold = float(getattr(args, 'plateau_loss_threshold', 1e-4))


def plateau_detector_state_matches(plateau_detector, state_dict):
    if plateau_detector is None or not isinstance(state_dict, dict):
        return False
    if state_dict.get('step_unit') != 'micro_step':
        return False

    requested_type = getattr(plateau_detector, 'detector_type', 'ema_improvement')
    saved_type = state_dict.get('detector_type')
    return saved_type == requested_type


def plateau_detector_log_fields(plateau_detector):
    if isinstance(plateau_detector, EmaImprovementPlateauDetector):
        return {
            'plateau/detector_code': 1,
            'plateau/ema_log_loss': plateau_detector.current_ema_log_loss(),
            'plateau/best_ema_log_loss': plateau_detector.best_log_loss(),
            'plateau/steps_since_best': plateau_detector.steps_since_best(),
            'plateau/last_improvement_log10': plateau_detector.last_improvement_log10(),
        }
    if isinstance(plateau_detector, EmaLossThresholdPlateauDetector):
        return {
            'plateau/detector_code': 2,
            'plateau/ema_loss': plateau_detector.current_ema_loss(),
            'plateau/loss_threshold': plateau_detector.loss_threshold,
        }
    return {}


def custom_get_scheduler(optimizer, num_training_steps, scheduler_type="inverse_sqrt",
                          num_warmup_steps=2000, cooldown_fraction=0.3):
    """Create a learning rate scheduler based on the specified type.

    Supported types:
        - 'inverse_sqrt': Inverse square root decay after warmup
        - 'linear': Linear decay from initial_lr to 0 after warmup
        - 'constant': Constant LR after warmup
        - 'cosine': Cosine decay from initial_lr to 0 after warmup
        - 'wsd': Warmup-Stable-Decay with externally-triggered cooldown
    """
    if scheduler_type == "linear":
        return LinearDecayScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "constant":
        return ConstantScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "cosine":
        return CosineScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "wsd":
        return WSDScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            max_training_steps=num_training_steps,
            cooldown_fraction=cooldown_fraction,
        )
    else:  # Default: inverse_sqrt
        return InverseSquareRootScheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )


def _evaluate_and_log_results(args, model, tokenizer, TO_TOKEN, step, cumulative_sample_tables, 
                             model_type, ema=None):
    """Helper function to evaluate model and return results for logging."""
    num_probs = getattr(args, 'num_eval_probs', 5)
    
    # Apply EMA if provided
    if ema is not None and model_type == "ema":
        ema.apply_ema(model)
    
    try:
        multi_prob_results = quick_evaluation_with_multiple_probs(
            args, model, tokenizer, TO_TOKEN, model_type, num_probs=num_probs, num_batches=16,
            cumulative_tables=cumulative_sample_tables, current_step=step, skip_tables=True
        )
        if multi_prob_results is None:
            multi_prob_results = quick_evaluation_iid(
                args, model, tokenizer, TO_TOKEN, model_type, num_batches=16,
                cumulative_tables=cumulative_sample_tables, current_step=step, skip_tables=True
            )
    finally:
        # Always restore original parameters if EMA was applied
        if ema is not None and model_type == "ema":
            ema.restore_original(model)
    
    return multi_prob_results


def _update_log_dict_with_results(log_dict, multi_prob_results, prefix=""):
    """Helper function to update log_dict with evaluation results."""
    if multi_prob_results is None:
        return
    
    # Log results for each probability or IID
    for switch_prob, prob_results in multi_prob_results.items():
        mean_str_acc = prob_results['mean_str_acc']
        mean_char_acc = prob_results['mean_char_acc']
        # Skip logging examples_table during training to avoid cluttering wandb
        # examples_table = prob_results.get('examples_table')

        if isinstance(switch_prob, str):
             # Use a generic name for string keys like 'iid' or 'addition'
             log_dict[f"{prefix}step_{switch_prob}_string_acc"] = mean_str_acc
             log_dict[f"{prefix}step_{switch_prob}_char_acc"] = mean_char_acc
        else:
             # Regular markov probability logging
             log_dict[f"{prefix}step_string_acc_prob_{switch_prob:.2f}"] = mean_str_acc
             log_dict[f"{prefix}step_char_acc_prob_{switch_prob:.2f}"] = mean_char_acc


def _get_checkpoint_dir(args):
    """Helper to get the checkpoint directory path."""
    base_save_path = os.path.join(args.checkpoint_root, args.wandb_project)
    checkpoint_dir = os.path.join(base_save_path, str(args.wandb_run_name))
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(args, accelerator, model, optimizer, scheduler, step, ema=None, plateau_detector=None):
    checkpoint_dir = _get_checkpoint_dir(args)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

    # Unwrap model for state dict when using accelerate
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model

    checkpoint_state = {
        'step': int(step),
        'model_state_dict': unwrapped.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
        'scheduler_type': getattr(args, 'scheduler_type', None),
        'scheduler_class': scheduler.__class__.__name__,
    }
    if ema is not None:
        checkpoint_state['ema_params'] = {k: v.clone().cpu() for k, v in ema.ema_params.items()}

    if plateau_detector is not None and hasattr(plateau_detector, 'state_dict'):
        checkpoint_state['plateau_detector'] = plateau_detector.state_dict()

    # Save wandb run ID for resumption
    if wandb.run is not None:
        checkpoint_state['wandb_run_id'] = wandb.run.id

    torch.save(checkpoint_state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at step {step}")


def save_training_complete_marker(args, reason: str, step: int):
    checkpoint_dir = _get_checkpoint_dir(args)
    marker_path = os.path.join(checkpoint_dir, "training_complete.txt")
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"reason={reason}\n")
            f.write(f"step={int(step)}\n")
            f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Training-complete marker saved to {marker_path}")
    except Exception as e:
        print(f"Warning: Failed to save training-complete marker to {marker_path}: {e}")


def save_models_in_run_dir(args, accelerator, model, ema=None):
    """Save latest and EMA (if available) models inside the run directory."""
    checkpoint_dir = _get_checkpoint_dir(args)

    # Unwrap model if using accelerate
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model

    # Save latest model
    latest_dir = os.path.join(checkpoint_dir, "latest")
    Path(latest_dir).mkdir(parents=True, exist_ok=True)
    if getattr(args, 'model', '') in ["mamba"]:
        torch.save(unwrapped, os.path.join(latest_dir, "model.pt"))
    else:
        if hasattr(unwrapped, 'save_pretrained'):
            unwrapped.save_pretrained(latest_dir)
        else:
            torch.save(unwrapped, os.path.join(latest_dir, "model.pt"))

    # Save EMA model if present
    if ema is not None and getattr(args, 'use_ema', False):
        ema_dir = os.path.join(checkpoint_dir, "ema")
        Path(ema_dir).mkdir(parents=True, exist_ok=True)
        # Use helper to temporarily apply ema params and save
        ema.save_ema_model(unwrapped, ema_dir)


def save_run_args(args):
    """Persist the training arguments alongside checkpoints for later test-time retrieval."""
    checkpoint_dir = _get_checkpoint_dir(args)
    args_path = os.path.join(checkpoint_dir, "args.json")
    try:
        # Convert Namespace to plain dict
        args_dict = {k: v for k, v in vars(args).items()}
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save args to {args_path}: {e}")


def _scheduler_state_looks_like_wsd(state_dict):
    return (
        isinstance(state_dict, dict)
        and (
            'max_training_steps' in state_dict
            or 'cooldown_fraction' in state_dict
            or 'cooldown_start' in state_dict
            or 'cooldown_length' in state_dict
        )
    )


def _should_load_scheduler_state(checkpoint, scheduler, requested_scheduler_type):
    state_dict = checkpoint.get('scheduler_state_dict') if isinstance(checkpoint, dict) else None
    if not state_dict:
        return False

    saved_scheduler_type = checkpoint.get('scheduler_type')
    saved_scheduler_class = checkpoint.get('scheduler_class')
    current_scheduler_class = scheduler.__class__.__name__

    if saved_scheduler_type is not None or saved_scheduler_class is not None:
        return (
            saved_scheduler_type == requested_scheduler_type
            or saved_scheduler_class == current_scheduler_class
        )

    # Legacy checkpoints did not store scheduler metadata. WSD state is
    # identifiable, so avoid loading it into a non-WSD scheduler.
    if _scheduler_state_looks_like_wsd(state_dict):
        return isinstance(scheduler, WSDScheduler)

    return not isinstance(scheduler, WSDScheduler)


def _restore_scheduler_progress(scheduler, state_dict):
    if not isinstance(state_dict, dict) or 'step_count' not in state_dict:
        return False

    progress_state = {
        'step_count': state_dict['step_count'],
    }
    for key in ('num_warmup_steps', 'initial_lr', 'min_lr'):
        if key in state_dict:
            progress_state[key] = state_dict[key]

    scheduler.load_state_dict(progress_state)
    return True


def collate_single_string(batch):
    return batch[0]

def train(args, model, train_dataset, TO_TOKEN, tokenizer, num_log_steps=200, max_runtime_seconds=0, start_time=None):
    optimizer = get_optimizer(model, args)
    
    # Set model to GPU
    from accelerate import Accelerator

    # Set mixed precision based on args
    mixed_precision = "no"
    simulate_fp8 = False
    if args.precision == "fp16":
        mixed_precision = "fp16"
    elif args.precision == "bf16":
        mixed_precision = "bf16"
    elif args.precision == "fp8":
        if torch.cuda.is_available():
            mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            mixed_precision = "no"
        simulate_fp8 = True
    accelerator = Accelerator(mixed_precision=mixed_precision)

    model, optimizer = accelerator.prepare(model, optimizer)

    if simulate_fp8:
        # Apply the FP8 simulation wrappers after accelerate prepare to ensure they target the right device parameters
        apply_fp8_simulation(model)
        print("Enabled simulated FP8 forward pass via mathematical truncation.")

    num_training_steps = args.steps
    steps_per_run = int(getattr(args, 'steps_per_run', 0) or 0)
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    # The scheduler should step based on actual optimizer update steps
    num_update_steps = num_training_steps // gradient_accumulation_steps

    lr_scheduler = custom_get_scheduler(
        optimizer,
        num_update_steps,
        args.scheduler_type,
        num_warmup_steps=getattr(args, 'num_warmup_steps', 2000),
        cooldown_fraction=getattr(args, 'cooldown_fraction', 0.2),
    )
    is_wsd_scheduler = isinstance(lr_scheduler, WSDScheduler)
    print(f"Using scheduler: {args.scheduler_type} ({lr_scheduler.__class__.__name__})")

    plateau_detector = None
    if is_wsd_scheduler:
        plateau_detector = make_plateau_detector(args)
        print(f"Using plateau detector: {getattr(args, 'plateau_detector', 'ema_improvement')}")

    # Track start time for time-based exit
    if start_time is None:
        start_time = time.time()
    time_limit_reached = False
    checkpoint_interval_seconds = int(getattr(args, 'checkpoint_interval_seconds', 0) or 0)
    checkpoint_interval_steps = int(getattr(args, 'checkpoint_interval_steps', 0) or 0)
    last_periodic_checkpoint_time = start_time
    last_periodic_checkpoint_step = 0

    # Resume from checkpoint if exists
    start_step = 1
    checkpoint_dir = _get_checkpoint_dir(args)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    loaded_checkpoint = None
    if os.path.exists(checkpoint_path):
        try:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Load model state
            accelerator.unwrap_model(model).load_state_dict(loaded_checkpoint['model_state_dict'])
            # Load optimizer state
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            # Load scheduler state if available
            if _should_load_scheduler_state(loaded_checkpoint, lr_scheduler, args.scheduler_type):
                lr_scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
                print("Scheduler state restored from checkpoint.")
            elif loaded_checkpoint.get('scheduler_state_dict'):
                saved_type = loaded_checkpoint.get('scheduler_type', 'unknown')
                saved_class = loaded_checkpoint.get('scheduler_class', 'unknown')
                if _restore_scheduler_progress(lr_scheduler, loaded_checkpoint['scheduler_state_dict']):
                    print(f"Scheduler type changed; restored shared scheduler progress only "
                          f"(saved type/class: {saved_type}/{saved_class}; "
                          f"current type/class: {args.scheduler_type}/{lr_scheduler.__class__.__name__}).")
                else:
                    print(f"Skipping scheduler state from checkpoint "
                          f"(saved type/class: {saved_type}/{saved_class}; "
                          f"current type/class: {args.scheduler_type}/{lr_scheduler.__class__.__name__}).")
            # Load plateau detector state if available. Older checkpoints stored
            # plateau steps in optimizer-update units; reset those buffers so
            # runtime CLI arguments now consistently mean training batch steps.
            if plateau_detector is not None and 'plateau_detector' in loaded_checkpoint and loaded_checkpoint['plateau_detector']:
                try:
                    plateau_state = loaded_checkpoint['plateau_detector']
                    if plateau_detector_state_matches(plateau_detector, plateau_state):
                        plateau_detector.load_state_dict(plateau_state)
                        print("Plateau detector state restored from checkpoint.")
                    else:
                        saved_detector_type = plateau_state.get('detector_type', 'legacy')
                        print(f"Plateau detector state is incompatible "
                              f"(saved type={saved_detector_type}, "
                              f"current type={getattr(args, 'plateau_detector', 'ema_improvement')}); "
                              f"resetting detector state.")
                except Exception as e:
                    print(f"Warning: Failed to restore plateau detector state: {e}")
            if plateau_detector is not None:
                refresh_plateau_detector_config(plateau_detector, args)
            start_step = int(loaded_checkpoint.get('step', 0)) + 1
            print(f"Resumed training from step {start_step}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}. Starting from scratch.")

    # Initialize EMA (optional)
    ema = None
    if getattr(args, 'use_ema', False):
        ema_beta = getattr(args, 'ema_beta', 0.999)
        ema = EMA(model, beta=ema_beta)
        print(f"Initialized EMA with beta={ema_beta}")
        # Restore EMA params if available
        if loaded_checkpoint is not None and 'ema_params' in loaded_checkpoint:
            ema.ema_params = {k: v.to(next(model.parameters()).device) for k, v in loaded_checkpoint['ema_params'].items()}
            print("EMA parameters restored from checkpoint.")
    

    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)

    avg_loss = [0]
    # Determine the end step for this run (if chunking is enabled)
    if steps_per_run > 0:
        end_step_this_run = min(start_step + steps_per_run - 1, num_training_steps)
    else:
        end_step_this_run = num_training_steps
    last_periodic_checkpoint_step = start_step - 1

    # Create DataLoader for parallel data generation
    num_workers = 2
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Dataset already returns a batch
        shuffle=False, # Dataset is random already
        num_workers=num_workers,
        collate_fn=collate_single_string, # Unwrap the list from batch_size=1
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )

    progress_bar = tqdm(
        enumerate(train_dataloader, start=start_step), 
        total=num_training_steps,
        desc='Training',
        initial=start_step - 1
    )

    import wandb
    # Initialize global cumulative tables for each probability
    cumulative_sample_tables = {}

    last_step_executed = start_step - 1
    for step, batch in progress_bar:
        if step > end_step_this_run:
            print(f"\nReached this run's step budget ({end_step_this_run}). Stopping to checkpoint.")
            break

        # Check if we've reached the target number of training steps (micro-steps)
        # This is critical when resuming from checkpoint to avoid running extra steps
        if step > num_training_steps:
            print(f"\nReached target training steps ({num_training_steps}). Stopping training.")
            break
        
        x = batch['input_ids'][:, :-1].to(device, non_blocking=True)
        y = batch['input_ids'][:, 1:].to(device, non_blocking=True)
        mask = batch['mask'][:, 1:].to(device, non_blocking=True)

        # Use efficient forward pass
        with accelerator.autocast():
            model_output = model(x)

        # Handle various output types to extract logits tensor
        if isinstance(model_output, dict):
            logits = model_output['logits']
        elif hasattr(model_output, 'logits'):
            logits = model_output.logits
        elif isinstance(model_output, (tuple, list)):
            logits = model_output[0]
        else:
            logits = model_output

        # Always compute loss manually to support focal loss and other custom losses
        actual_logits = logits

        loss = ce_loss(y, actual_logits, mask, TO_TOKEN)
        

            
        
        # Log average loss, curriculum information, sample table every num_log_steps
        # We accumulate loss just for logging purposes to get an average over the accumulation steps
        if 'accumulated_loss_for_log' not in locals():
            accumulated_loss_for_log = 0.0
            micro_steps_in_log_window = 0
            
        accumulated_loss_for_log += actual_logits.detach().float().new_tensor(loss.item()) # Use tensor to keep on device if needed or just float
        micro_steps_in_log_window += 1

        # Scale loss for gradient accumulation (so accumulated gradients are averaged, not summed)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            lr_scheduler.step()
                
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA parameters after each optimization step (optional)
            if ema is not None and getattr(args, 'use_ema', False):
                ema.update(model)
            
            # Check time limit after each optimizer step
            if max_runtime_seconds > 0:
                elapsed_seconds = time.time() - start_time
                if elapsed_seconds >= max_runtime_seconds:
                    print(f"\nTime limit reached ({elapsed_seconds:.0f}s >= {max_runtime_seconds}s). Saving checkpoint and exiting...")
                    time_limit_reached = True
                    last_step_executed = step
                    save_checkpoint(args, accelerator, model, optimizer, lr_scheduler, step, ema, plateau_detector)
                    save_run_args(args)
                    save_models_in_run_dir(args, accelerator, model, ema)
                    break

            periodic_checkpoint_reasons = []
            now = time.time()
            if checkpoint_interval_seconds > 0:
                seconds_since_checkpoint = now - last_periodic_checkpoint_time
                if seconds_since_checkpoint >= checkpoint_interval_seconds:
                    periodic_checkpoint_reasons.append(f"{seconds_since_checkpoint:.0f}s elapsed")
            if checkpoint_interval_steps > 0:
                steps_since_checkpoint = step - last_periodic_checkpoint_step
                if steps_since_checkpoint >= checkpoint_interval_steps:
                    periodic_checkpoint_reasons.append(f"{steps_since_checkpoint} steps elapsed")

            if periodic_checkpoint_reasons:
                print(f"\nPeriodic checkpoint at step {step} ({'; '.join(periodic_checkpoint_reasons)}). "
                      "Overwriting checkpoint.pt...")
                try:
                    save_checkpoint(args, accelerator, model, optimizer, lr_scheduler, step, ema, plateau_detector)
                    save_run_args(args)
                except Exception as e:
                    print(f"Warning: Failed to save periodic checkpoint at step {step}: {e}")
                last_periodic_checkpoint_time = now
                last_periodic_checkpoint_step = step
            
            # --- LOGGING AND EVALUATION ---
            # Now defined based on update steps
            current_update_step = step // gradient_accumulation_steps
            
            if current_update_step % num_log_steps == 0:
                # Calculate avg loss over the accumulation window
                if micro_steps_in_log_window > 0:
                    avg_loss_val = accumulated_loss_for_log.item() / micro_steps_in_log_window
                else:
                    avg_loss_val = 0.0
                accumulated_loss_for_log = 0.0 # Reset
                micro_steps_in_log_window = 0

                avg_loss.append(avg_loss_val)
                memory_mb = get_memory_usage()
                log_dict = {
                    "train_loss": avg_loss_val,
                    "step": current_update_step,
                    "micro_step": step,
                    "gpu_memory_mb": memory_mb,
                    "lr": optimizer.param_groups[0]['lr'],
                    "weight_decay": optimizer.param_groups[0].get('weight_decay', 0.0),
                }

                if is_wsd_scheduler and plateau_detector is not None:
                    # Plateau detection uses raw training batch/micro steps so its
                    # CLI step arguments share the same unit as --steps.
                    current_plateau_step = step
                    plateau_triggered = plateau_detector.update(current_plateau_step, avg_loss_val)
                    log_dict['lr/phase'] = lr_scheduler.phase()
                    log_dict['plateau/batch_step'] = current_plateau_step
                    log_dict.update(plateau_detector_log_fields(plateau_detector))

                    # Trigger WSD cooldown on first detected plateau
                    if plateau_triggered and lr_scheduler.cooldown_start is None:
                        lr_scheduler.start_cooldown()
                        if isinstance(plateau_detector, EmaImprovementPlateauDetector):
                            detector_detail = (
                                f"ema_log_loss={plateau_detector.current_ema_log_loss():.6f}, "
                                f"steps_since_best={plateau_detector.steps_since_best()}"
                            )
                        else:
                            detector_detail = (
                                f"ema_loss={plateau_detector.current_ema_loss():.6f}, "
                                f"threshold={plateau_detector.loss_threshold:.6f}"
                            )
                        print(f"\nPlateau detected at batch step {current_plateau_step} "
                              f"(update step {current_update_step}, "
                              f"loss={avg_loss_val:.6f}, {detector_detail}). "
                              f"Starting WSD cooldown over {lr_scheduler.cooldown_length} steps.")
                        log_dict['plateau/cooldown_triggered_step'] = current_plateau_step
                        log_dict['plateau/cooldown_triggered_batch_step'] = current_plateau_step
                        log_dict['plateau/cooldown_triggered_update_step'] = current_update_step
                        log_dict['plateau/cooldown_triggered_loss'] = avg_loss_val
                        log_dict['plateau/cooldown_triggered_detector_code'] = (
                            1 if isinstance(plateau_detector, EmaImprovementPlateauDetector) else 2
                        )
                        # Refresh phase after transition so it reflects the new state
                        log_dict['lr/phase'] = lr_scheduler.phase()

                # Log memory usage to console
                log_memory_usage(current_update_step, f"Training (Update {current_update_step}) ")
                

                # Evaluation with multiple switch probabilities
                model_was_training = model.training
                model.eval()
                
                with torch.no_grad():
                    # Evaluate latest model
                    latest_results = _evaluate_and_log_results(
                        args, model, tokenizer, TO_TOKEN, current_update_step, cumulative_sample_tables, "latest"
                    )
                    _update_log_dict_with_results(log_dict, latest_results, "latest_")

                    # Evaluate EMA model (optional)
                    if ema is not None:
                        ema_results = _evaluate_and_log_results(
                            args, model, tokenizer, TO_TOKEN, current_update_step, cumulative_sample_tables, "ema", ema
                        )
                        _update_log_dict_with_results(log_dict, ema_results, "ema_")
                
                if model_was_training:
                    model.train()
                
                wandb.log(log_dict, step=current_update_step)

                progress_bar.set_postfix({'Loss': avg_loss_val, 'Step': current_update_step})

            # WSD-done exit path: cooldown finished, training is complete.
            if isinstance(lr_scheduler, WSDScheduler) and lr_scheduler.is_done():
                print(f"\nWSD cooldown complete at step {step} "
                      f"(update step {step // gradient_accumulation_steps}). Saving final checkpoint and exiting loop.")
                last_step_executed = step
                try:
                    save_checkpoint(args, accelerator, model, optimizer, lr_scheduler, step, ema, plateau_detector)
                    save_run_args(args)
                    save_models_in_run_dir(args, accelerator, model, ema)
                    save_training_complete_marker(args, "wsd_cooldown_complete", step)
                except Exception as e:
                    print(f"Warning: Failed to save final checkpoint on WSD-done: {e}")
                break

        # Track the last executed step (updated after every micro-step, but used for checkpointing)
        last_step_executed = step

    # Save final checkpoint after loop completion
    try:
        print(f"Loop finished. Saving checkpoint at step {last_step_executed}.")
        save_checkpoint(args, accelerator, model, optimizer, lr_scheduler, last_step_executed, ema, plateau_detector)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint after run: {e}")

    # Persist args and save final latest/EMA snapshots inside the run directory
    try:
        save_run_args(args)
        save_models_in_run_dir(args, accelerator, model, ema)
    except Exception as e:
        print(f"Warning: Failed to save final models/args in run dir: {e}")

    return ema, time_limit_reached


