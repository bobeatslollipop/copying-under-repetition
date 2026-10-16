"""API clients, samplers, and scoring from the original LLM experiment runner."""

import requests
import random
import string
import json
import numpy as np
import time
import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional
from datetime import datetime
from pathlib import Path


# Literal prompt delimiters; subword tokenization depends on the provider.

BOS = "$"
EOS = "."
CPY = ":"

SYSTEM_PROMPT = (
    "You are a precise string-copying assistant. "
    f"You will see a string enclosed between '{BOS}' and '{EOS}', "
    f"followed by '{CPY}'. "
    "Output ONLY the enclosed string, character by character, with absolutely "
    "nothing else -- no spaces before or after, no explanation, no newline."
)


# ── Data Generation ──

def rand_str_iid(length: int, vocab_size: int = 26) -> str:
    chars = list(string.ascii_lowercase[:vocab_size])
    return ''.join(random.choices(chars, k=length))


def rand_str_markov(length: int, switch_prob: float,
                    vocab_size: int = 26,
                    letters_pair: Optional[tuple] = None) -> tuple[str, tuple]:
    all_chars = list(string.ascii_lowercase[:vocab_size])
    if letters_pair is None:
        letters_pair = tuple(random.sample(all_chars, 2))
    a, b = letters_pair
    current = random.choice([a, b])
    chars = [current]
    for _ in range(length - 1):
        if random.random() < switch_prob:
            current = b if current == a else a
        chars.append(current)
    return ''.join(chars), letters_pair


def rand_str_k_letter_geometric(length: int, vocab_size: int = 26, L: float = 0.0) -> str:
    if length == 0:
        return ""
    alphabet = list(string.ascii_lowercase[:vocab_size])
    if L <= 0:
        chars = np.random.choice(alphabet, size=length)
        return ''.join(chars)
    if L == float('inf'):
        char = np.random.choice(alphabet)
        return char * length
    p_switch = 1.0 / (L + 1.0)
    chars = []
    remaining = length
    while remaining > 0:
        char = np.random.choice(alphabet)
        run_len = min(np.random.geometric(p_switch), remaining)
        chars.append(char * run_len)
        remaining -= run_len
    return ''.join(chars)


# ── Prompt Construction ──

def build_prompt(target: str) -> str:
    return f"{BOS}{target}{EOS}  {CPY}"


def build_few_shot_prompt(target: str, n_shots: int = 3) -> str:
    lines = []
    for _ in range(n_shots):
        ex = rand_str_iid(random.randint(6, 12))
        lines.append(f"{BOS}{ex}{EOS}  {CPY} {ex}")
    lines.append(f"{BOS}{target}{EOS}  {CPY}")
    return "\n".join(lines)


# ── API Clients ──

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.last_response_metadata = {}

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def _post(self, endpoint: str, payload: dict, timeout: int) -> tuple[str, str, str]:
        """Returns (content_text, error_msg, thinking_text)."""
        payload["stream"] = True
        content_text = ""
        thinking_text = ""
        error_msg = ""
        saw_done = False
        self.last_response_metadata = {}
        try:
            with requests.post(f"{self.base_url}/api/{endpoint}",
                               json=payload, timeout=timeout, stream=True) as resp:
                self.last_response_metadata["http_status"] = resp.status_code
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    for key in ("model", "created_at", "done", "done_reason", "total_duration",
                                "load_duration", "prompt_eval_count", "prompt_eval_duration",
                                "eval_count", "eval_duration"):
                        if key in chunk:
                            self.last_response_metadata[key] = chunk[key]
                    if chunk.get("error"):
                        self.last_response_metadata["stream_error"] = str(chunk["error"])
                        error_msg = f"Ollama stream error: {chunk['error']}"
                        break
                    if endpoint == "chat":
                        msg = chunk.get("message", {})
                        # CoT: 'reasoning_content' (DeepSeek/Qwen3) or 'thinking' (Qwen3.5)
                        for key in ("reasoning_content", "thinking"):
                            if msg.get(key):
                                thinking_text += msg[key]
                        if msg.get("content"):
                            content_text += msg["content"]
                    else:
                        if chunk.get("response"):
                            content_text += chunk["response"]
                    if chunk.get("done"):
                        saw_done = True
                        reason = chunk.get("done_reason")
                        if reason and reason != "stop":
                            error_msg = f"Ollama done_reason={reason}"
                        break
                if not saw_done and not error_msg:
                    error_msg = "Ollama stream ended without a completion marker"
        except Exception as e:
            error_msg = str(e)
            self.last_response_metadata["request_error_type"] = type(e).__name__
        return content_text, error_msg, thinking_text

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        payload = {
            "model": model, "prompt": prompt, "system": system,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": max_tokens + 10000,
                        "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0},
        }
        if think is not None:
            payload["think"] = think
        return self._post("generate", payload, timeout)

    def chat(self, model: str, user_message: str, system: str = "",
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        payload = {
            "model": model, "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": max_tokens + 10000,
                        "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0},
        }
        if think is not None:
            payload["think"] = think
        return self._post("chat", payload, timeout)

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
            return True
        except Exception:
            return False


class OpenAIResponsesClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 reasoning_effort: str = "none", max_retries: int = 8,
                 retry_initial_sleep: float = 1.0, retry_max_sleep: float = 300.0):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_initial_sleep = retry_initial_sleep
        self.retry_max_sleep = retry_max_sleep

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _extract_text(self, data: dict) -> str:
        if data.get("output_text"):
            return data["output_text"]
        parts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        return "".join(parts)

    def _post_responses(self, payload: dict, timeout: int) -> tuple[str, str, str]:
        content_text = ""
        error_msg = ""
        retry_sleep = self.retry_initial_sleep

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/responses",
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{resp.status_code} Server Error: {resp.text[:500]}",
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content_text = self._extract_text(data)
                error_msg = ""
                if data.get("status") == "incomplete":
                    details = data.get("incomplete_details") or {}
                    error_msg = f"incomplete response: {details.get('reason', 'unknown')}"
                return content_text, error_msg, ""
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                error_msg = str(e)
                if attempt >= self.max_retries:
                    break
                print(
                    f"OpenAI request failed on attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{error_msg}. Sleeping {retry_sleep:.1f}s before retry.",
                    flush=True,
                )
                time.sleep(retry_sleep)
                retry_sleep = min(retry_sleep * 2, self.retry_max_sleep)
            except Exception as e:
                error_msg = str(e)
                break

        return content_text, error_msg, ""

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        return self.chat(model, prompt, system, max_tokens, temperature, timeout, think)

    def chat(self, model: str, user_message: str, system: str = "",
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        payload = {
            "model": model,
            "instructions": system,
            "input": user_message,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": self.reasoning_effort},
        }
        return self._post_responses(payload, timeout)


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 thinking: str = "disabled", max_retries: int = 8,
                 retry_initial_sleep: float = 1.0, retry_max_sleep: float = 300.0):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.thinking = thinking
        self.max_retries = max_retries
        self.retry_initial_sleep = retry_initial_sleep
        self.retry_max_sleep = retry_max_sleep

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _extract_text(self, data: dict) -> tuple[str, str]:
        content_text = ""
        thinking_text = ""
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("reasoning_content"):
                thinking_text += msg["reasoning_content"]
            if msg.get("content"):
                content_text += msg["content"]
        return content_text, thinking_text

    def _post_chat_completions(self, payload: dict, timeout: int) -> tuple[str, str, str]:
        content_text = ""
        thinking_text = ""
        error_msg = ""
        retry_sleep = self.retry_initial_sleep

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code in {408, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{resp.status_code} Server Error: {resp.text[:500]}",
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content_text, thinking_text = self._extract_text(data)
                error_msg = ""
                for choice in data.get("choices", []):
                    finish_reason = choice.get("finish_reason", "")
                    if finish_reason and finish_reason not in {"stop"}:
                        error_msg = f"DeepSeek finish_reason={finish_reason}"
                        break
                return content_text, error_msg, thinking_text
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                error_msg = str(e)
                if attempt >= self.max_retries:
                    break
                print(
                    f"DeepSeek request failed on attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{error_msg}. Sleeping {retry_sleep:.1f}s before retry.",
                    flush=True,
                )
                time.sleep(retry_sleep)
                retry_sleep = min(retry_sleep * 2, self.retry_max_sleep)
            except Exception as e:
                error_msg = str(e)
                break

        return content_text, error_msg, thinking_text

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        return self.chat(model, prompt, system, max_tokens, temperature, timeout, think)

    def chat(self, model: str, user_message: str, system: str = "",
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        thinking_type = self.thinking
        if think is False:
            thinking_type = "disabled"
        elif think is True:
            thinking_type = "enabled"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": thinking_type},
        }
        return self._post_chat_completions(payload, timeout)


class GeminiClient:
    def __init__(self, api_key: str,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 thinking_level: str = "low", thinking_budget: int | None = None,
                 max_retries: int = 8, retry_initial_sleep: float = 1.0,
                 retry_max_sleep: float = 300.0):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.retry_initial_sleep = retry_initial_sleep
        self.retry_max_sleep = retry_max_sleep

    @property
    def headers(self) -> dict:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _model_path(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _extract_text(self, data: dict) -> tuple[str, str]:
        content_text = ""
        thinking_text = ""
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text") or ""
                if part.get("thought"):
                    thinking_text += text
                else:
                    content_text += text
        return content_text, thinking_text

    def _post_generate_content(self, model: str, payload: dict,
                               timeout: int) -> tuple[str, str, str]:
        content_text = ""
        thinking_text = ""
        error_msg = ""
        retry_sleep = self.retry_initial_sleep
        url = f"{self.base_url}/{self._model_path(model)}:generateContent"

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{resp.status_code} Server Error: {resp.text[:500]}",
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content_text, thinking_text = self._extract_text(data)
                error_msg = ""
                for candidate in data.get("candidates", []):
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason and finish_reason != "STOP":
                        error_msg = f"Gemini finishReason={finish_reason}"
                        break
                if not data.get("candidates") and data.get("promptFeedback"):
                    error_msg = f"Gemini promptFeedback={data['promptFeedback']}"
                return content_text, error_msg, thinking_text
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                error_msg = str(e)
                if attempt >= self.max_retries:
                    break
                print(
                    f"Gemini request failed on attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{error_msg}. Sleeping {retry_sleep:.1f}s before retry.",
                    flush=True,
                )
                time.sleep(retry_sleep)
                retry_sleep = min(retry_sleep * 2, self.retry_max_sleep)
            except Exception as e:
                error_msg = str(e)
                break

        return content_text, error_msg, thinking_text

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.0,
                 timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        return self.chat(model, prompt, system, max_tokens, temperature, timeout, think)

    def chat(self, model: str, user_message: str, system: str = "",
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: int = 1200, think: bool | None = None) -> tuple[str, str, str]:
        thinking_config = {}
        if self.thinking_budget is not None:
            thinking_config["thinkingBudget"] = self.thinking_budget
        elif self.thinking_level:
            thinking_config["thinkingLevel"] = self.thinking_level

        generation_config = {
            "candidateCount": 1,
            "temperature": temperature,
            "topP": 1.0,
            "maxOutputTokens": max_tokens,
        }
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": user_message}],
            }],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return self._post_generate_content(model, payload, timeout)


# ── Output Extraction & Metrics ──

def extract_output(raw_response: str, target: str = "") -> str:
    """Extract the copy from model output.

    Strategies (in order):
    A. Find lines that are purely [a-z]+ after cleaning -> pick closest to target length
    B. Fallback: clean the last non-empty line
    """
    out = raw_response.strip()
    # Remove <think>...</think> blocks (or unclosed <think> if truncated)
    out = re.sub(r"<think>.*?(</think>|$)", "", out, flags=re.DOTALL).strip()

    lines = [l.strip() for l in out.split('\n') if l.strip()]
    if not lines:
        return ""

    def _clean(s):
        for art in ["Output:", "output:", "Copy:", "copy:", "Answer:", "answer:",
                     "Result:", "result:", BOS, EOS, CPY, "`", '"', "'"]:
            s = s.replace(art, "")
        s = s.replace(" ", "")  # collapse "a b c" -> "abc"
        return s.strip()

    # Strategy A: pure lowercase lines
    candidates = [c for l in lines if (c := _clean(l)) and re.fullmatch(r'[a-z]+', c)]
    if candidates:
        if target:
            candidates.sort(key=lambda c: abs(len(c) - len(target)))
        return candidates[0] if target else candidates[-1]

    # Strategy B: fallback
    return _clean(lines[-1])


def char_accuracy(pred: str, target: str) -> float:
    """Calculates accuracy based on Normalized Levenshtein Distance.
    
    This handles cases where the model inserts or deletes characters, 
    preserving accuracy for subsequent matching segments.
    Accuracy = 1 - (edit_distance / max(len(pred), len(target)))
    """
    if not target:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    
    n_t, n_p = len(target), len(pred)
    
    # Optimization: ensure the smaller string is target
    if n_p < n_t:
        pred, target = target, pred
        n_p, n_t = n_t, n_p
    
    # Distance O(N) space DP
    prev_row = list(range(n_t + 1))
    for i in range(1, n_p + 1):
        curr_row = [i]
        for j in range(1, n_t + 1):
            sub_cost = 0 if pred[i-1] == target[j-1] else 1
            curr_row.append(min(
                prev_row[j] + 1,       # deletion
                curr_row[j-1] + 1,     # insertion
                prev_row[j-1] + sub_cost # substitution
            ))
        prev_row = curr_row
    
    dist = prev_row[n_t]
    return max(0.0, 1.0 - dist / n_p)


def string_accuracy(pred: str, target: str) -> float:
    return float(pred == target)


def length_penalty(pred: str, target: str) -> float:
    if not target:
        return 1.0
    return 1.0 - abs(len(pred) - len(target)) / len(target)


# ── Trial & Experiment ──

@dataclass
class TrialResult:
    model: str
    condition: str
    string_length: int
    switch_prob: float
    trial_idx: int
    target: str
    raw_output: str         # full output: thinking + content (for debugging)
    prediction: str
    char_acc: float
    str_acc: float
    len_penalty: float
    latency_s: float
    error: str = ""
    L_value: float = -1.0


def run_trial(client: OllamaClient, model: str, target: str,
              condition: str, switch_prob: float, L_value: float, trial_idx: int,
              use_chat_api: bool = True, few_shot: bool = False,
              verbose: bool = False, think: bool | None = None,
              max_output_tokens: int = 20000) -> TrialResult:
    prompt_text = build_few_shot_prompt(target) if few_shot else build_prompt(target)
    t0 = time.time()
    error = ""
    content = ""
    thinking = ""

    try:
        max_tokens = max_output_tokens
        call = client.chat if use_chat_api else client.generate
        kwargs = dict(
            model=model, system=SYSTEM_PROMPT, max_tokens=max_tokens, think=think,
        )
        if use_chat_api:
            kwargs["user_message"] = prompt_text
        else:
            kwargs["prompt"] = prompt_text
        content, error, thinking = call(**kwargs)
    except Exception as e:
        error = error or str(e)

    latency = time.time() - t0

    if not error and not content.strip():
        error = "empty output"
    prediction = "" if error else extract_output(content, target)

    # raw_output for CSV: full thinking + content combined
    # Even if an error/timeout occurs, the partial content is saved!
    full_raw = ""
    if thinking:
        full_raw += f"<think>{thinking}</think>\n"
    full_raw += content
    raw_csv = full_raw.replace('\r\n', '\\n').replace('\r', '').replace('\n', '\\n')

    result = TrialResult(
        model=model, condition=condition, string_length=len(target),
        switch_prob=switch_prob, trial_idx=trial_idx, target=target,
        raw_output=raw_csv, prediction=prediction,
        char_acc=char_accuracy(prediction, target),
        str_acc=string_accuracy(prediction, target),
        len_penalty=length_penalty(prediction, target),
        latency_s=round(latency, 2), error=error, L_value=L_value,
    )

    if verbose:
        status = "OK" if result.str_acc == 1.0 else "FAIL"
        src = "empty" if error == "empty output" else ("error" if error else "content")
        print(f"  {status} [{condition}] len={len(target)} "
              f"char_acc={result.char_acc:.3f} str_acc={result.str_acc:.0f} "
              f"({latency:.1f}s) src={src}")
        if error:
            print(f"    ERROR: {error}")
        elif result.str_acc < 1.0 and len(target) <= 30:
            print(f"    target: {target}")
            print(f"    pred  : {prediction[:80]}")

    return result


@dataclass
class ExperimentConfig:
    models: list[str]
    lengths: list[int]
    switch_probs: list[float] = None   # 0.0 = i.i.d.
    L_values: list[float] = None
    n_trials: int = 20
    use_chat_api: bool = True
    few_shot: bool = False
    think: bool | None = None   # None=default, False=disable CoT
    run_name: Optional[str] = None
    output_dir: str = "results"
    verbose: bool = True
    resume: bool = False
    time_limit_s: int = 0
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_reasoning_effort: str = "none"
    openai_max_retries: int = 8
    openai_retry_initial_sleep: float = 1.0
    openai_retry_max_sleep: float = 300.0
    gemini_api_key: str = ""
    gemini_thinking_level: str = "low"
    gemini_thinking_budget: int | None = None
    gemini_max_retries: int = 8
    gemini_retry_initial_sleep: float = 1.0
    gemini_retry_max_sleep: float = 300.0
    deepseek_api_key: str = ""
    deepseek_thinking: str = "disabled"
    deepseek_max_retries: int = 8
    deepseek_retry_initial_sleep: float = 1.0
    deepseek_retry_max_sleep: float = 300.0
    max_output_tokens: int = 20000


def condition_name(switch_prob: float) -> str:
    return "iid" if switch_prob == 0.0 else f"markov_p={switch_prob:.2f}"

def condition_name_L(L_value: float) -> str:
    if L_value == 0.0:
        return "iid"
    elif L_value == float('inf'):
        return "geometric_L=inf"
    else:
        return f"geometric_L={L_value}"


def run_experiment(cfg: ExperimentConfig) -> list[TrialResult]:
    start_time = time.time()
    if cfg.provider == "openai":
        client = OpenAIResponsesClient(
            api_key=cfg.openai_api_key,
            base_url=cfg.base_url,
            reasoning_effort=cfg.openai_reasoning_effort,
            max_retries=cfg.openai_max_retries,
            retry_initial_sleep=cfg.openai_retry_initial_sleep,
            retry_max_sleep=cfg.openai_retry_max_sleep,
        )
    elif cfg.provider == "gemini":
        client = GeminiClient(
            api_key=cfg.gemini_api_key,
            base_url=cfg.base_url,
            thinking_level=cfg.gemini_thinking_level,
            thinking_budget=cfg.gemini_thinking_budget,
            max_retries=cfg.gemini_max_retries,
            retry_initial_sleep=cfg.gemini_retry_initial_sleep,
            retry_max_sleep=cfg.gemini_retry_max_sleep,
        )
    elif cfg.provider == "deepseek":
        client = DeepSeekClient(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.base_url,
            thinking=cfg.deepseek_thinking,
            max_retries=cfg.deepseek_max_retries,
            retry_initial_sleep=cfg.deepseek_retry_initial_sleep,
            retry_max_sleep=cfg.deepseek_retry_max_sleep,
        )
    else:
        client = OllamaClient(base_url=cfg.base_url)
    if not client.is_available():
        if cfg.provider == "openai":
            raise RuntimeError("OpenAI API key is empty. Set OPENAI_API_KEY.")
        if cfg.provider == "gemini":
            raise RuntimeError("Gemini API key is empty. Set GEMINI_API_KEY.")
        if cfg.provider == "deepseek":
            raise RuntimeError("DeepSeek API key is empty. Set DEEPSEEK_API_KEY.")
        raise RuntimeError(f"Ollama is not running at {cfg.base_url}. Start it with: OLLAMA_HOST=... ollama serve")

    try:
        available = client.list_models()
        print(f"Available models at {cfg.base_url}: {available}\n")
        for m in cfg.models:
            if not any(m in a for a in available):
                if cfg.provider in {"openai", "gemini", "deepseek"}:
                    print(f"WARNING: '{m}' not found in /models response; continuing anyway.")
                else:
                    print(f"WARNING: '{m}' not found. Pull with: ollama pull {m}")
    except Exception as e:
        print(f"WARNING: Could not list models at {cfg.base_url}: {e}. Continuing anyway.")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    if cfg.run_name:
        csv_filename = f"{cfg.run_name}.csv"
    else:
        csv_filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(cfg.output_dir, csv_filename)
    response_log_path = Path(csv_path).with_suffix(".responses.jsonl")
    if cfg.provider == "ollama" and response_log_path.exists():
        if not cfg.resume:
            raise FileExistsError(f"Refusing to overwrite response log {response_log_path}")
        if response_log_path.stat().st_size:
            with response_log_path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("Existing response log ends with an incomplete line; inspect it before resuming")

    all_results: list[TrialResult] = []
    existing_keys = set()
    if cfg.resume and os.path.exists(csv_path):
        if os.path.getsize(csv_path):
            with open(csv_path, "rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) not in (b"\n", b"\r"):
                    raise ValueError("Existing CSV ends with an incomplete line; inspect it before resuming")
            all_results = load_results_from_csv(csv_path, strict=True)
        for r in all_results:
            key = (r.model, r.condition, r.string_length, r.trial_idx)
            if key in existing_keys:
                raise ValueError(f"Duplicate recorded trial: {key}")
            existing_keys.add(key)
        print(f"Resuming experiment. Preserving all {len(existing_keys)} recorded trials in {csv_path}.")

    num_conds = (len(cfg.switch_probs) if cfg.switch_probs else 0) + (len(cfg.L_values) if cfg.L_values else 0)
    total = len(cfg.models) * len(cfg.lengths) * num_conds * cfg.n_trials
    done = len(existing_keys)

    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode = "a" if cfg.resume else "x"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialResult.__dataclass_fields__))
        if not has_header:
            writer.writeheader()
            f.flush()

        for model in cfg.models:
            print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

            conditions_to_run = []
            if cfg.switch_probs:
                for sp in cfg.switch_probs:
                    conditions_to_run.append(("switch_prob", sp))
            if cfg.L_values:
                for L in cfg.L_values:
                    conditions_to_run.append(("L_value", L))

            for c_type, c_val in conditions_to_run:
                if c_type == "switch_prob":
                    cond = condition_name(c_val)
                    sp = c_val
                    L = -1.0
                else:
                    cond = condition_name_L(c_val)
                    sp = -1.0
                    L = c_val

                for length in cfg.lengths:
                    print(f"\n  Condition={cond}, Length={length}")

                    for trial_idx in range(cfg.n_trials):
                        if cfg.time_limit_s > 0 and (time.time() - start_time) > cfg.time_limit_s:
                            print(f"\nTime limit of {cfg.time_limit_s}s reached. Exiting with code 99 to trigger resubmission.")
                            sys.exit(99)

                        if c_type == "switch_prob":
                            if sp == 0.0:
                                target = rand_str_iid(length)
                            else:
                                target, _ = rand_str_markov(length, sp)
                        else:
                            target = rand_str_k_letter_geometric(length, 26, L)

                        # Consume the same sampler draws when resuming with a fixed seed.
                        if (model, cond, length, trial_idx) in existing_keys:
                            continue

                        result = run_trial(
                            client, model, target, cond, sp, L, trial_idx,
                            cfg.use_chat_api, cfg.few_shot, cfg.verbose, cfg.think,
                            cfg.max_output_tokens,
                        )
                        all_results.append(result)

                        if cfg.provider == "ollama":
                            response_record = {
                                "recorded_at": datetime.now().astimezone().isoformat(),
                                "model": model, "condition": cond, "string_length": length,
                                "trial_idx": trial_idx,
                                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                                "error": result.error, "response": client.last_response_metadata,
                            }
                            with response_log_path.open("a", encoding="utf-8") as response_log:
                                response_log.write(json.dumps(response_record) + "\n")
                        row = asdict(result)
                        writer.writerow(row)
                        f.flush()
                        done += 1

                    subset = [r for r in all_results
                              if r.model == model and r.condition == cond
                              and r.string_length == length]
                    print(f"  >> avg char_acc={np.mean([r.char_acc for r in subset]):.3f}  "
                          f"avg str_acc={np.mean([r.str_acc for r in subset]):.3f}  "
                          f"({done}/{total} done)")

    print(f"\nResults saved to: {csv_path}")
    return all_results


# ── Analysis ──

def summarize(results: list[TrialResult]) -> dict:
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        raw[r.model][r.condition][r.string_length].append(r)

    out = {}
    for model, conds in raw.items():
        out[model] = {}
        for cond, lengths in conds.items():
            out[model][cond] = {
                l: {"char_acc": np.mean([t.char_acc for t in ts]),
                    "str_acc": np.mean([t.str_acc for t in ts]), "n": len(ts)}
                for l, ts in sorted(lengths.items()) if len(ts) > 0
            }
    return out


def print_summary_table(summary: dict):
    print(f"\n{'='*70}\nSUMMARY TABLE (String Accuracy)\n{'='*70}")
    for model, conds in summary.items():
        print(f"\nModel: {model}")
        all_lengths = sorted({l for c in conds.values() for l in c})
        header = f"{'Condition':<25}" + "".join(f"{l:>8}" for l in all_lengths)
        print(header)
        print("-" * len(header))
        for cond, lengths in sorted(conds.items()):
            row = f"{cond:<25}"
            for l in all_lengths:
                row += f"{lengths[l]['str_acc']:>8.3f}" if l in lengths else f"{'N/A':>8}"
            print(row)






def load_results_from_csv(path: str, strict: bool = False) -> list[TrialResult]:
    csv.field_size_limit(sys.maxsize)
    results = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, strict=strict)
        if strict and reader.fieldnames != list(TrialResult.__dataclass_fields__):
            raise ValueError(f"Unexpected CSV columns in {path}")
        for row in reader:
            try:
                if strict and (None in row or any(value is None for value in row.values())):
                    raise ValueError("Incomplete CSV row")
                results.append(TrialResult(
                    model=row["model"], condition=row["condition"],
                    string_length=int(row["string_length"]),
                    switch_prob=float(row["switch_prob"]),
                    trial_idx=int(row["trial_idx"]), target=row["target"],
                    raw_output=row["raw_output"], prediction=row["prediction"],
                    char_acc=float(row["char_acc"]), str_acc=float(row["str_acc"]),
                    len_penalty=float(row["len_penalty"]),
                    latency_s=float(row.get("latency_s") or 0.0),
                    error=row.get("error", ""),
                    L_value=float(row.get("L_value") if row.get("L_value") else -1.0),
                ))
            except (ValueError, KeyError, TypeError) as error:
                if strict:
                    raise ValueError(f"Invalid recorded trial at CSV line {reader.line_num}") from error
                continue
    return results
