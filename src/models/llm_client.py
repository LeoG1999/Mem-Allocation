"""
Unified LLM client for GPT-4o, o1-mini, o3-mini, and DeepSeek.
Key features:
- records thinking_tokens for reasoning models
- provides one interface across providers
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0        # o1/o3 reasoning tokens
    latency_ms: float = 0.0
    model: str = ""
    raw: Any = field(default=None, repr=False)

    @property
    def visible_answer_tokens(self) -> int:
        """
        Approximate number of visible answer tokens.

        This is an answer-side observation, not a definition of usable context.
        Reasoning tokens quantify reasoning overhead but are not an input budget.
        """
        return max(0, self.output_tokens - self.thinking_tokens)


# Model configuration table
MODEL_CONFIGS = {
    # Standard LLM
    "gpt-4o": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "context_window": 128_000,
        "is_reasoning": False,
    },
    "gpt-4o-mini": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "context_window": 128_000,
        "is_reasoning": False,
    },
    # Reasoning models
    "o1-mini": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "context_window": 128_000,
        "is_reasoning": True,
    },
    "o3-mini": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "context_window": 200_000,
        "is_reasoning": True,
    },
    # DeepSeek (OpenAI-compatible)
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window": 64_000,
        "is_reasoning": False,
    },
    "deepseek-v4-flash": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window": 1_000_000,
        "is_reasoning": False,
        "api_model_env": "DEEPSEEK_V4_FLASH_MODEL_ID",
        "api_model": "deepseek-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "deepseek-reasoner": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "context_window": 64_000,
        "is_reasoning": True,
    },
    # Together.ai — Llama 3 8B Instruct Lite (serverless, OpenAI-compatible)
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "context_window": 8_000,
        "is_reasoning": False,
    },
    # Together.ai — Qwen 3.5-9B (serverless, OpenAI-compatible, 262K context).
    # Qwen3.5 is a hybrid reasoning model with thinking mode on by default; we
    # disable it via chat_template_kwargs for direct answer generation.
    "Qwen/Qwen3.5-9B": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "context_window": 262_144,
        "is_reasoning": False,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

# Semantic aliases let experiment scripts name V3 non-reasoning and reasoning modes.
# They still map to the official DeepSeek API model names.
MODEL_ALIASES = {
    "deepseek-v3": "deepseek-chat",
    "deepseek-v3-nonthinking": "deepseek-chat",
    "deepseek-v3-thinking": "deepseek-reasoner",
    "deepseek-v3-reasoning": "deepseek-reasoner",
}


class LLMClient:
    def __init__(self, model: str = "gpt-4o"):
        resolved_model = MODEL_ALIASES.get(model, model)
        if resolved_model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model}. Supported: {list(MODEL_CONFIGS)}")
        self.model = resolved_model
        self.requested_model = model
        cfg = MODEL_CONFIGS[resolved_model]
        self.api_key_env = cfg["api_key_env"]
        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(
                f"Missing API key: set {cfg['api_key_env']} before running model {resolved_model}"
            )
        model_env = cfg.get("api_model_env")
        deployment = os.environ.get(model_env) if model_env else None
        self.api_model = deployment or cfg.get("api_model", resolved_model)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("base_url"),
        )
        self.context_window = cfg["context_window"]
        self.is_reasoning = cfg["is_reasoning"]
        self.default_extra_body = cfg.get("extra_body")

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        **kwargs,
    ) -> LLMResponse:
        """Call synchronously and return LLMResponse, retrying transient API errors."""
        t0 = time.perf_counter()

        call_kwargs: Dict[str, Any] = {"model": self.api_model, "messages": messages}
        # Avoid indefinite hangs on provider side; callers can override via kwargs.
        call_kwargs["timeout"] = kwargs.pop("timeout", 60)

        if self.is_reasoning:
            # o1/o3 use max_completion_tokens and do not support temperature=0.
            call_kwargs["max_completion_tokens"] = max_tokens
        else:
            call_kwargs["temperature"] = temperature
            call_kwargs["max_tokens"] = max_tokens

        # Inject model-specific extra_body (e.g. Qwen3.5 enable_thinking=False)
        if self.default_extra_body is not None:
            user_extra = kwargs.pop("extra_body", {}) or {}
            merged = {**self.default_extra_body, **user_extra}
            call_kwargs["extra_body"] = merged

        call_kwargs.update(kwargs)

        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**call_kwargs)
                break
            except openai.AuthenticationError as e:
                raise RuntimeError(
                    f"Authentication failed for model={self.model} ({self.api_key_env}). "
                    "Please rotate/update the API key."
                ) from e
            except openai.APIConnectionError as e:
                last_exc = e
                wait = 2 ** attempt * 2
                time.sleep(wait)
            except openai.RateLimitError as e:
                last_exc = e
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                time.sleep(wait)
            except openai.APIStatusError as e:
                if e.status_code in (500, 502, 503):
                    last_exc = e
                    time.sleep(2 ** attempt * 2)
                else:
                    raise
        else:
            raise last_exc

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage

        # thinking tokens:
        #   o1/o3: usage.completion_tokens_details.reasoning_tokens
        #   deepseek-reasoner: same field, or reasoning_content on the message
        thinking_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            thinking_tokens = getattr(
                usage.completion_tokens_details, "reasoning_tokens", 0
            ) or 0
        # DeepSeek fallback: estimate from message.reasoning_content length.
        if thinking_tokens == 0 and self.is_reasoning:
            rc = getattr(resp.choices[0].message, "reasoning_content", None)
            if rc:
                thinking_tokens = len(rc) // 4  # Character-to-token estimate

        result = LLMResponse(
            content=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            thinking_tokens=thinking_tokens,
            latency_ms=latency_ms,
            model=self.model,
            raw=None,   # Do not retain SDK objects with cyclic references.
        )
        del resp        # Release the ChatCompletion object so GC can reclaim it promptly.
        return result
