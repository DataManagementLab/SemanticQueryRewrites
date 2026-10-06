########################################################################################################################
# LLM API Helpers Version: 2025-11-25
########################################################################################################################
import abc
import collections
import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import multiprocessing
import multiprocessing.managers
import multiprocessing.pool
import os
import pathlib
import threading
import time
import typing

import requests
import tqdm

logger = logging.getLogger(__name__)

CACHE_PATH: pathlib.Path = pathlib.Path("llm_cache")
BUDGET: float | None = None  # budget per model provider, `None` means always execute!

MODELS: dict[str, "ModelInfo"] = {
    ####################################################################################################################
    # OpenAI: see https://platform.openai.com/docs/models and https://platform.openai.com/docs/pricing
    ####################################################################################################################
    # GPT 5.4
    "gpt-5.4-2026-03-05": {  # added 2026-03-21
        "provider": "openai",
        "max_context": 1_050_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {
            "default": 2.5,
            "batch": 1.25,
            "flex": 1.25,
            "priority": 5,
        },
        "usd_per_1m_cached_input_tokens": {
            "default": 0.25,
            "batch": 0.13,
            "flex": 0.13,
            "priority": 0.5,
        },
        "usd_per_1m_output_tokens": {
            "default": 15,
            "batch": 7.5,
            "flex": 7.5,
            "priority": 30.00,
        },
        "openai_tokenizer_encoding_name": "o200k_base",
    },
    
    # GPT-5-Codex
    "gpt-5-codex": {  # added 2025-11-11
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {"default": 1.25, "priority": 2.50},
        "usd_per_1m_cached_input_tokens": {"default": 0.125, "priority": 0.25},
        "usd_per_1m_output_tokens": {"default": 10.00, "priority": 20.00},
        "openai_tokenizer_encoding_name": "o200k_base",
        "model_api": "response"  # TODO: implement OpenAI responses API as its own provider
    },
    "gpt-5.1-codex": {  # added 2025-11-17
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {"default": 1.25, "priority": 2.50},
        "usd_per_1m_cached_input_tokens": {"default": 0.125, "priority": 0.25},
        "usd_per_1m_output_tokens": {"default": 10.00, "priority": 20.00},
        "openai_tokenizer_encoding_name": "o200k_base",
        "model_api": "response"  # TODO: implement OpenAI responses API as its own provider
    },

    # GPT-5.1
    "gpt-5.1-2025-11-13": {  # added 2025-11-20
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {
            "default": 1.25,
            "batch": 0.625,
            "flex": 0.625,
            "priority": 2.50,
        },
        "usd_per_1m_cached_input_tokens": {
            "default": 0.125,
            "batch": 0.0625,
            "flex": 0.0625,
            "priority": 0.25,
        },
        "usd_per_1m_output_tokens": {
            "default": 10.00,
            "batch": 5.00,
            "flex": 5.00,
            "priority": 20.00,
        },
        "openai_tokenizer_encoding_name": "o200k_base",
    },

    # GPT-5
    "gpt-5-2025-08-07": {  # added 2025-08-08
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {"default": 1.25, "batch": 0.625, "flex": 0.625, "priority": 2.50},
        "usd_per_1m_cached_input_tokens": {"default": 0.125, "batch": 0.0625, "flex": 0.0625, "priority": 0.25},
        "usd_per_1m_output_tokens": {"default": 10.00, "batch": 5.00, "flex": 5.00, "priority": 20.00},
        "openai_tokenizer_encoding_name": "o200k_base"
    },
    "gpt-5-mini-2025-08-07": {  # added 2025-08-08
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {"default": 0.25, "batch": 0.125, "flex": 0.125, "priority": 0.45},
        "usd_per_1m_cached_input_tokens": {"default": 0.025, "batch": 0.0125, "flex": 0.0125, "priority": 0.05},
        "usd_per_1m_output_tokens": {"default": 2.00, "batch": 1.00, "flex": 1.00, "priority": 3.60},
        "openai_tokenizer_encoding_name": "o200k_base"
    },
    "gpt-5-nano-2025-08-07": {  # added 2025-08-08
        "provider": "openai",
        "max_context": 400_000,
        "max_output_tokens": 128_000,
        "usd_per_1m_input_tokens": {"default": 0.05, "batch": 0.025, "flex": 0.025},
        "usd_per_1m_cached_input_tokens": {"default": 0.005, "batch": 0.0025, "flex": 0.0025},
        "usd_per_1m_output_tokens": {"default": 0.40, "batch": 0.20, "flex": 0.20},
        "openai_tokenizer_encoding_name": "o200k_base"
    },

    # GPT-4.1
    "gpt-4.1-2025-04-14": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 1_047_576,
        "max_output_tokens": 32_768,
        "usd_per_1m_input_tokens": {"default": 2.00, "batch": 1.00, "flex": 1.00},
        "usd_per_1m_cached_input_tokens": {"default": 0.50, "batch": 0.50, "flex": 0.50},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 8.00, "batch": 4.00, "flex": 1.00},
        "openai_tokenizer_encoding_name": "o200k_base"
    },
    "gpt-4.1-mini-2025-04-14": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 1_047_576,
        "max_output_tokens": 32_768,
        "usd_per_1m_input_tokens": {"default": 0.40, "batch": 0.20, "flex": 0.20},
        "usd_per_1m_cached_input_tokens": {"default": 0.10, "batch": 0.10, "flex": 0.10},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 1.60, "batch": 0.80, "flex": 0.80},
        "openai_tokenizer_encoding_name": "o200k_base"
    },
    "gpt-4.1-nano-2025-04-14": {  # added 2025-04-30  # https://github.com/openai/tiktoken/issues/395
        "provider": "openai",
        "max_context": 1_047_576,
        "max_output_tokens": 32_768,
        "usd_per_1m_input_tokens": {"default": 0.10, "batch": 0.05, "flex": 0.05},
        "usd_per_1m_cached_input_tokens": {"default": 0.025, "batch": 0.025, "flex": 0.025},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 0.40, "batch": 0.20, "flex": 0.20},
        "openai_tokenizer_encoding_name": "o200k_base"
    },

    # GPT-4o
    "gpt-4o-2024-11-20": {  # added 2025-04-30  # https://github.com/openai/tiktoken/issues/395
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 2.50, "batch": 1.25, "flex": 1.25},
        "usd_per_1m_cached_input_tokens": {"default": 1.25, "batch": 1.25, "flex": 1.25},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 10.00, "batch": 5.00, "flex": 5.00}
    },
    "gpt-4o-2024-08-06": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 2.50, "batch": 1.25, "flex": 1.25},
        "usd_per_1m_cached_input_tokens": {"default": 1.25, "batch": 1.25, "flex": 1.25},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 10.00, "batch": 5.00, "flex": 5.00}
    },
    "gpt-4o-2024-05-13": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 5.00, "batch": 2.50, "flex": 2.50},
        "usd_per_1m_cached_input_tokens": {},
        "usd_per_1m_output_tokens": {"default": 15.00, "batch": 7.50, "flex": 7.50}
    },
    "gpt-4o-mini-2024-07-18": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 0.15, "batch": 0.075, "flex": 0.075},
        "usd_per_1m_cached_input_tokens": {"default": 0.075, "batch": 0.075, "flex": 0.075},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 0.60, "batch": 0.30, "flex": 0.30}
    },

    # o1, o1-pro, o3, o4-mini, o3-mini, o-mini
    "o1-2024-12-17": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 15.00, "batch": 7.50, "flex": 30.00},
        "usd_per_1m_cached_input_tokens": {"default": 7.50, "batch": 7.50, "flex": 7.50},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 60.00, "batch": 7.50, "flex": 30.00}
    },
    "o1-preview-2024-09-12": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 15.00, "batch": 7.50, "flex": 30.00},
        "usd_per_1m_cached_input_tokens": {"default": 7.50, "batch": 7.50, "flex": 7.50},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 60.00, "batch": 7.50, "flex": 30.00}
    },
    "o1-pro-2025-03-19": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 150.00, "batch": 75.00, "flex": 75.00},
        "usd_per_1m_cached_input_tokens": {},
        "usd_per_1m_output_tokens": {"default": 600.00, "batch": 300.00, "flex": 300.00}
    },
    "o3-2025-04-16": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 10.00, "batch": 5.00, "flex": 5.00},
        "usd_per_1m_cached_input_tokens": {"default": 2.50, "batch": 2.50, "flex": 2.50},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 40.00, "batch": 20.00, "flex": 20.00}
    },
    "o4-mini-2025-04-16": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 1.10, "batch": 0.55, "flex": 0.55},
        "usd_per_1m_cached_input_tokens": {"default": 0.275, "batch": 0.275, "flex": 0.275},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 4.40, "batch": 2.20, "flex": 2.20}
    },
    "o3-mini-2025-01-31": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 200_000,
        "max_output_tokens": 100_000,
        "usd_per_1m_input_tokens": {"default": 1.10, "batch": 0.55, "flex": 0.55},
        "usd_per_1m_cached_input_tokens": {"default": 0.55, "batch": 0.55, "flex": 0.55},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 4.40, "batch": 2.20, "flex": 2.20}
    },
    "o1-mini-2024-09-12": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 65_536,
        "usd_per_1m_input_tokens": {"default": 1.10, "batch": 0.55, "flex": 0.55},
        "usd_per_1m_cached_input_tokens": {"default": 0.55, "batch": 0.55, "flex": 0.55},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 4.40, "batch": 2.20, "flex": 2.20}
    },

    # Other models
    "gpt-3.5-turbo-0125": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 16_385,
        "max_output_tokens": 4_096,
        "usd_per_1m_input_tokens": {"default": 0.50, "batch": 0.25},
        "usd_per_1m_cached_input_tokens": {},
        "usd_per_1m_output_tokens": {"default": 1.50, "batch": 0.75}
    },

    ####################################################################################################################
    # Ollama: see https://ollama.com
    ####################################################################################################################
    # Llama 3.1
    "llama3.1:8b-instruct-fp16": {  # added 2025-04-30
        "provider": "ollama",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_context": 128_000  # this must also be configured in Ollama
    },
    "llama3.1:70b-instruct-fp16": {  # added 2025-05-13
        "provider": "ollama",
        "hf_model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "max_context": 128_000  # this must also be configured in Ollama
    },

    ####################################################################################################################
    # vLLM: see https://docs.vllm.ai/en/latest/models/supported_models.html
    ####################################################################################################################
    "meta-llama/Llama-3.1-8B-Instruct": {  # added 2025-06-30
        "provider": "vllm",
        "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct"
    },
    "meta-llama/Llama-3.1-70B-Instruct": {  # added 2025-07-01
        "provider": "vllm",
        "hf_model_name": "meta-llama/Llama-3.1-70B-Instruct",
    },

    ####################################################################################################################
    # Anthropic: see https://docs.claude.com/en/docs/about-claude/models/overview and https://claude.com/pricing#api
    ####################################################################################################################
    "claude-sonnet-4-5-20250929": {  # added 2025-11-12
        "provider": "anthropic",
        "max_context": 200_000,  # there is a beta with 1M context window
        "max_output_tokens": 64_000,
        "usd_per_1m_input_tokens": {
            "default": 3,
            "default_cache_write": 3.75,
            "prompt_longer_than_200K": 6,
            "prompt_longer_than_200K_cache_write": 7.50
        },
        "usd_per_1m_cached_input_tokens": {
            "default": 0.30,
            "prompt_longer_than_200K": 0.60
        },
        "usd_per_1m_output_tokens": {
            "default": 15.00,
            "prompt_longer_than_200K": 22.50
        }
    },
    "claude-opus-4-5-20251101": {  # added 2025-11-25
        "provider": "anthropic",
        "max_context": 200_000,  # there is a beta with 1M context window
        "max_output_tokens": 64_000,
        "usd_per_1m_input_tokens": {
            "default": 5,
            "default_cache_write": 6.25,
        },
        "usd_per_1m_cached_input_tokens": {
            "default": 0.50
        },
        "usd_per_1m_output_tokens": {
            "default": 25.00
        }
    },
}

NUM_EXECUTION_THREADS: int = 300
NUM_TIKTOKEN_PROCESSES: int = 16

RETRY_ON_HTTP_STATUS_CODES: list[int] = [503]
MAX_TRIES: int = 3  # rate limit errors always lead to retry

PROVIDERS: dict[str, "BaseProvider"] = {}

type Request = dict
type Response = dict


def execute(
        request_or_requests: Request | list[Request],
        /, *,
        budget: float | None = 0.0,
        use_cache: bool = True,
        silent: bool = False,
        auto_transform: typing.Literal["openai", "ollama", "vllm"] | None = None
) -> Response | list[Response]:
    """Execute one request synchronously or multiple requests in parallel.

    Args:
        request_or_requests: The request or list of requests to execute.
        budget: Execute without confirmation if costs are below budget. `None` means always execute!
        use_cache: Load/store responses for identical requests from/in the cache.
        silent: Disable log messages and progress bars.
        auto_transform: Automatically transform requests and responses.

    Returns:
        The response or list of responses.
    """
    if request_or_requests == []:
        return []

    provider = BaseProvider.check_determine_provider(request_or_requests)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    request_or_requests = provider.transform_request_or_requests_from(request_or_requests, auto_provider)

    if isinstance(request_or_requests, dict):  # dict = Request
        response_or_responses = provider.execute([request_or_requests], budget, use_cache, silent, None)[0]
    elif isinstance(request_or_requests, list):
        if not silent:
            with ProgressBar(total=len(request_or_requests)) as progress_bar:
                response_or_responses = provider.execute(request_or_requests, budget, use_cache, silent, progress_bar)
        else:
            response_or_responses = provider.execute(request_or_requests, budget, use_cache, silent, None)
    else:
        typing.assert_never(request_or_requests)
    return provider.transform_response_or_responses_to(response_or_responses, auto_provider)


def count_tokens(
        request_or_requests: Request | list[Request],
        /, *,
        silent: bool = False,
        auto_transform: typing.Literal["openai", "ollama", "vllm"] | None = None
) -> int | list[int]:
    """Count the number of input tokens for one request synchronously or for multiple requests in parallel.

    Args:
        request_or_requests: The request or list of requests for which to count input tokens.
        silent: Disable log messages and progress bars.
        auto_transform: Automatically transform requests.

    Returns:
        The number of tokens or list of numbers of tokens.
    """
    if request_or_requests == []:
        return []

    provider = BaseProvider.check_determine_provider(request_or_requests)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    request_or_requests = provider.transform_request_or_requests_from(request_or_requests, auto_provider)

    if isinstance(request_or_requests, dict):  # dict = Request
        return provider.count_tokens([request_or_requests], silent, None)[0]
    elif isinstance(request_or_requests, list):
        if not silent:
            with ProgressBar(total=len(request_or_requests)) as progress_bar:
                return provider.count_tokens(request_or_requests, silent, progress_bar)
        else:
            return provider.count_tokens(request_or_requests, silent, None)
    else:
        typing.assert_never(request_or_requests)


def count_tokens_str(
        string_or_strings: str | list[str],
        model: str,
        /, *,
        silent: bool = False
) -> int | list[int]:
    """Count the number of input tokens for one string synchronously or for multiple strings in parallel.

    Args:
        string_or_strings: The string or list of strings for which to count tokens.
        model: The name of the model.
        silent: Disable log messages and progress bars.

    Returns:
        The number of tokens or list of numbers of tokens.
    """
    if string_or_strings == []:
        return []

    provider = BaseProvider.check_determine_provider(model)

    if isinstance(string_or_strings, str):
        return provider.count_tokens_str([string_or_strings], model, silent, None)[0]
    elif isinstance(string_or_strings, list):
        if not silent:
            with ProgressBar(total=len(string_or_strings)) as progress_bar:
                return provider.count_tokens_str(string_or_strings, model, silent, progress_bar)
        else:
            return provider.count_tokens_str(string_or_strings, model, silent, None)
    else:
        typing.assert_never(string_or_strings)


def cost(
        response_or_responses: Response | list[Response],
        /, *,
        ignore_token_caching: bool = False,
        silent: bool = False,
        auto_transform: typing.Literal["openai", "ollama", "vllm"] | None = None
) -> float | list[float]:
    """Compute the USD costs for one or multiple responses.

    Args:
        response_or_responses: The response or list of responses.
        ignore_token_caching: Ignore that some tokens were cached and therefore cheaper.
        silent: Disable log messages and progress bars.
        auto_transform: Automatically transform responses.

    Returns:
        The USD cost or list of costs.
    """
    if response_or_responses == []:
        return []

    provider = BaseProvider.check_determine_provider(response_or_responses)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    if auto_provider is not None:
        response_or_responses = auto_provider.transform_response_or_responses_to(response_or_responses, provider)

    if isinstance(response_or_responses, dict):  # dict = Response
        return provider.cost([response_or_responses], ignore_token_caching, silent, None)[0]
    elif isinstance(response_or_responses, list):
        if not silent:
            with ProgressBar(total=len(response_or_responses)) as progress_bar:
                return provider.cost(response_or_responses, ignore_token_caching, silent, progress_bar)
        else:
            return provider.cost(response_or_responses, ignore_token_caching, silent, None)
    else:
        typing.assert_never(response_or_responses)


@contextlib.contextmanager
def client():
    """Client context manager to orchestrate API calls when using multiprocessing."""
    with multiprocessing.Manager() as manager:
        for provider in PROVIDERS.values():
            provider.state.migrate(manager.dict(), manager.Semaphore(), manager.Semaphore())
        try:
            yield
        finally:
            for provider in PROVIDERS.values():
                provider.state.migrate({}, threading.Semaphore(), threading.Semaphore())


########################################################################################################################
# generic implementation
########################################################################################################################


class ModelInfo(typing.TypedDict, total=False):
    provider: str
    max_context: int
    max_output_tokens: int
    usd_per_1m_input_tokens: dict[str, float]
    usd_per_1m_cached_input_tokens: dict[str, float]
    usd_per_1m_output_tokens: dict[str, float]
    hf_model_name: str
    openai_tokenizer_encoding_name: str  # optional, used instead of model name to find tokenizer
    # TODO: implement OpenAI responses API as its own provider
    model_api: typing.Literal["completion", "response"]  # optional, default is "completion"


@dataclasses.dataclass
class State:
    state: dict | multiprocessing.managers.DictProxy
    state_semaphore: threading.Semaphore
    wait_semaphore: threading.Semaphore

    def migrate(
            self,
            state: dict | multiprocessing.managers.DictProxy,
            state_semaphore: threading.Semaphore,
            wait_semaphore: threading.Semaphore
    ) -> None:
        with state_semaphore:
            with self.state_semaphore:
                for k, v in self.state.items():
                    state[k] = v
                self.state = state
                self.state_semaphore = state_semaphore
                self.wait_semaphore = wait_semaphore


class ProgressBar(tqdm.tqdm):
    cached: int
    running: int
    failed: int
    cost: float
    bottleneck: typing.Literal["P", "S", "L"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cached = 0
        self.failed = 0
        self.cost = 0
        self.bottleneck = "P"
        self.update_postfix()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return super().__exit__(exc_type, exc_value, traceback)

    def next_step(self, desc: str, left: int) -> None:
        super().reset(total=self.total)
        self.set_description(desc)
        self.update(self.total - left)

    def update(self, *args, **kwargs) -> None:
        self.update_postfix()
        super().update(*args, **kwargs)

    def update_postfix(self) -> None:
        bottleneck = None if self.bottleneck == "P" else self.bottleneck
        failed = f"failed={self.failed}" if self.failed > 0 else None
        cached = f"cached={self.cached}" if self.cached > 0 else None
        cost_ = f"cost=${self.cost:.2f}" if self.cost > 0 else None

        self.set_postfix_str(", ".join(p for p in (bottleneck, failed, cached, cost_) if p is not None))


@dataclasses.dataclass
class BasePair(abc.ABC):
    request: Request
    response: Response | None = None
    num_tries: int = 0


class BaseProvider(abc.ABC):
    name: str = "base-provider"
    trans_warning_cache: set
    state: State

    def __init__(self) -> None:
        self.trans_warning_cache = set()
        self.state = State({}, threading.Semaphore(), threading.Semaphore())

    @classmethod
    def check_determine_model(cls, value: str | Request | list[Request] | Response | list[Response]) -> str:
        if isinstance(value, str):
            model = value
        elif isinstance(value, dict):  # dict = Request | Response
            model = value["model"]
        elif isinstance(value, list):
            models = set(r["model"] for r in value)
            assert len(models) == 1, "all requests/responses must use the same model"
            (model,) = models
        else:
            typing.assert_never(value)
        assert model in MODELS.keys(), f"unknown model `{model}`"
        return model

    @classmethod
    def check_determine_provider(cls, value: str | Request | list[Request] | Response | list[Response]) -> typing.Self:
        model = cls.check_determine_model(value)
        provider = MODELS[model]["provider"]
        assert provider in PROVIDERS.keys(), f"unknown model provider `{provider}`"
        return PROVIDERS[provider]

    @classmethod
    def check_determine_auto_provider(cls, auto_transform: str | None) -> typing.Self | None:
        if auto_transform is not None:
            assert auto_transform in PROVIDERS.keys(), f"unknown model provider `{auto_transform}` for auto_transform"
            return PROVIDERS[auto_transform]
        return None

    def transform_request_or_requests_from(
            self,
            request_or_requests: Request | list[Request],
            from_provider: typing.Self | None
    ) -> Request | list[Request]:
        if from_provider is None:
            return request_or_requests
        elif isinstance(request_or_requests, dict):  # dict = Request
            return self.transform_request_from(request_or_requests, from_provider)
        elif isinstance(request_or_requests, list):
            return [self.transform_request_from(req, from_provider) for req in request_or_requests]
        else:
            typing.assert_never(request_or_requests)

    def transform_request_from(self, request: Request, from_provider: typing.Self) -> Request:
        match from_provider.name:
            case "openai":
                return self.transform_request_from_openai_to_self(request)
            case "ollama":
                return self.transform_request_from_ollama_to_self(request)
            case "vllm":
                return self.transform_request_from_vllm_to_self(request)
            case "anthropic":
                return self.transform_request_from_anthropic_to_self(request)
            case _ as other:
                typing.assert_never(other)

    def transform_response_or_responses_to(
            self,
            response_or_responses: Response | list[Response],
            to_provider: typing.Self | None
    ) -> Response | list[Response]:
        if to_provider is None:
            return response_or_responses
        elif isinstance(response_or_responses, dict):  # dict = Response
            return self.transform_response_to(response_or_responses, to_provider)
        elif isinstance(response_or_responses, list):
            return [self.transform_response_to(res, to_provider) for res in response_or_responses]
        else:
            typing.assert_never(response_or_responses)

    def transform_response_to(self, response: Response, to_provider: typing.Self) -> Response:
        match to_provider.name:
            case "openai":
                return self.transform_response_from_self_to_openai(response)
            case "ollama":
                return self.transform_response_from_self_to_ollama(response)
            case "vllm":
                return self.transform_response_from_self_to_vllm(response)
            case "anthropic":
                return self.transform_response_from_self_to_anthropic(response)
            case _ as other:
                typing.assert_never(other)

    def user_confirms_cost(self, cost_: float, budget: float, silent: bool, progress_bar: ProgressBar | None) -> None:
        if progress_bar is not None:
            progress_bar.clear()
        with self.state.state_semaphore:
            total_cost = self.state.state["total_cost"]
            if budget is not None and cost_ > budget or BUDGET is not None and total_cost + cost_ > BUDGET:
                message = f"press enter to spend up to around ${cost_:.2f}"
                if total_cost > 0:
                    message = f"already spent ${total_cost:.2f}, {message} more"
                logger.info(message)
                # Deactivate verfication of prompt cost
                # input(message)
            elif not silent:
                message = f"spending up to around ${cost_:.2f}"
                if total_cost > 0:
                    message = f"already spent ${total_cost:.2f}, now {message} more"
                logger.info(message)
            self.state.state["total_cost"] = self.state.state["total_cost"] + cost_

    def compute_hash(self, pair: BasePair, prefix: str) -> str:
        return hashlib.sha256(bytes(f"{prefix}-{self.name}-{json.dumps(pair.request)}", "utf-8")).hexdigest()

    def load_cached_response(self, pair: BasePair, prefix: str, use_cache: bool) -> bool:
        if use_cache:
            path = CACHE_PATH / f"{self.compute_hash(pair, prefix)}.json"
            if path.is_file():
                with open(path, "r", encoding="utf-8") as file:
                    cached_pair = json.load(file)
                if cached_pair["provider"] == self.name and cached_pair["request"] == pair.request:
                    pair.response = cached_pair["response"]
                    return True
        return False

    def save_to_cache(self, pair: BasePair, prefix: str, use_cache: bool) -> None:
        if use_cache:
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
            path = CACHE_PATH / f"{self.compute_hash(pair, prefix)}.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump({
                    "provider": self.name,
                    "request": pair.request,
                    "response": pair.response
                }, file)

    def trans_warning(self, message: str) -> None:
        message_prefix = message[:message.rindex("=")] if "=" in message else message
        if message_prefix not in self.trans_warning_cache:
            logger.warning(message)
            self.trans_warning_cache.add(message_prefix)

    @staticmethod
    def autodict() -> collections.defaultdict:
        return collections.defaultdict(BaseProvider.autodict)

    @classmethod
    def autodict_to_dict(cls, x: typing.Any) -> typing.Any:
        if isinstance(x, dict):
            return {k: cls.autodict_to_dict(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [cls.autodict_to_dict(v) for v in x]
        else:
            return x

    @abc.abstractmethod
    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            use_cache: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        raise NotImplementedError()

    @abc.abstractmethod
    def count_tokens(self, requests: list[Request], silent: bool, progress_bar: ProgressBar | None) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_request_from_openai_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_request_from_ollama_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_request_from_vllm_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_request_from_anthropic_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_response_from_self_to_openai(self, response: Response) -> Response:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_response_from_self_to_ollama(self, response: Response) -> Response:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_response_from_self_to_vllm(self, response: Response) -> Response:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_response_from_self_to_anthropic(self, response: Response) -> Response:
        raise NotImplementedError()


########################################################################################################################
# OpenAI implementation
########################################################################################################################


@dataclasses.dataclass
class OpenAIRateLimitBudget:
    mode: typing.Literal["sequential", "parallel"] = "sequential"
    rpm: float | None = None
    tpm: float | None = None
    r: float | None = None
    t: float | None = None
    last_update: float = dataclasses.field(default_factory=time.time)

    def is_enough_for_request(self, pair: "OpenAIPair") -> bool:
        return (self.r is None or self.r >= 1) and (self.t is None or self.t >= pair.est_max_usage)

    def consider_time(self) -> typing.Self:
        now = time.time()
        delta = now - self.last_update
        if self.rpm is not None and self.r is not None:
            self.r = min(self.rpm, self.r + self.rpm * delta / 60)
        if self.tpm is not None and self.t is not None:
            self.t = min(self.tpm, self.t + self.tpm * delta / 60)
        self.last_update = now
        return self

    def decrease_by_request(self, pair: "OpenAIPair") -> typing.Self:
        if self.r is not None:
            self.r -= 1
        if self.t is not None:
            self.t -= pair.est_max_usage
        return self

    def set_from_headers(self, headers: dict[str, typing.Any]) -> typing.Self:
        if "x-ratelimit-limit-requests" in headers.keys():
            self.rpm = int(headers["x-ratelimit-limit-requests"])
        if "x-ratelimit-limit-project-requests" in headers.keys():
            self.rpm = int(headers["x-ratelimit-limit-project-requests"])
        if "x-ratelimit-limit-tokens" in headers.keys():
            self.tpm = int(headers["x-ratelimit-limit-tokens"])
        if "x-ratelimit-limit-project-tokens" in headers.keys():
            self.tpm = int(headers["x-ratelimit-limit-project-tokens"])
        if "x-ratelimit-remaining-requests" in headers.keys():
            header_r = int(headers["x-ratelimit-remaining-requests"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        if "x-ratelimit-remaining-project-requests" in headers.keys():
            header_r = int(headers["x-ratelimit-remaining-project-requests"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        if "x-ratelimit-remaining-tokens" in headers.keys():
            header_t = int(headers["x-ratelimit-remaining-tokens"])
            if self.t is None or self.t > header_t:
                self.t = header_t
        if "x-ratelimit-remaining-project-tokens" in headers.keys():
            header_t = int(headers["x-ratelimit-remaining-project-tokens"])
            if self.t is None or self.t > header_t:
                self.t = header_t
        return self

    def to_par(self) -> typing.Self:
        self.mode = "parallel"
        return self

    def to_seq(self) -> typing.Self:
        self.mode = "sequential"
        return self


@dataclasses.dataclass
class OpenAIPair(BasePair):
    est_max_cost: float | None = None
    est_max_usage: int | None = None

    def set_cost_and_usage(self, est_input_tokens: int) -> None:
        model_params = MODELS[self.request["model"]]

        assert "n" not in self.request.keys() and "best_of" not in self.request.keys(), "`n` and `best_of` not supported"

        if "max_completion_tokens" in self.request.keys() and self.request["max_completion_tokens"] is not None:
            est_max_output_tokens = self.request["max_completion_tokens"]
        elif "max_tokens" in self.request.keys() and self.request["max_tokens"] is not None:
            est_max_output_tokens = self.request["max_tokens"]
        else:
            left_for_output = max(0, model_params["max_context"] - est_input_tokens)
            est_max_output_tokens = min(left_for_output, model_params["max_output_tokens"])

        mode = "default"
        if "service_tier" in self.request.keys():
            mode = self.request["service_tier"]
        self.est_max_cost = est_input_tokens * model_params["usd_per_1m_input_tokens"][mode] / 1_000_000 \
                            + est_max_output_tokens * model_params["usd_per_1m_output_tokens"][mode] / 1_000_000
        self.est_max_usage = est_input_tokens + est_max_output_tokens

    def execute(
            self,
            provider: "OpenAIProvider",
            progress_bar: ProgressBar | None,
            use_cache: bool,
            has_semaphore: bool
    ) -> bool:
        # TODO: implement OpenAI responses API as its own provider
        if MODELS[self.request["model"]].get("model_api","completion") == "response":
            url = "https://api.openai.com/v1/responses"
            # Transform request for Response API: 'messages' -> 'input'
            request_payload = {k: v for k, v in self.request.items() if k not in ["messages","max_completion_tokens"]}
            request_payload["input"] = self.request["messages"]
            request_payload["max_output_tokens"] = self.request["max_completion_tokens"]
        else:
            url = "https://api.openai.com/v1/chat/completions"
            request_payload = self.request

        http_response = requests.post(
            url=url,
            json=request_payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        )
        self.num_tries += 1

        with provider.state.state_semaphore if not has_semaphore else contextlib.nullcontext():
            provider.state.state["total_calls"] = provider.state.state["total_calls"] + 1
            provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].set_from_headers(
                http_response.headers
            )
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self, "execute-request", use_cache)
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_par()
                    cost_ = provider.cost([self.response], False, True, None)[0]
                    provider.state.state["total_cost"] = provider.state.state["total_cost"] + cost_ - self.est_max_cost
                    if progress_bar is not None:
                        progress_bar.cost += cost_
                        progress_bar.update()
                    return True
                case 429:
                    logger.info("retry request due to rate limit error")
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_seq()
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case status_code if status_code in RETRY_ON_HTTP_STATUS_CODES and self.num_tries < MAX_TRIES:
                    logger.info(f"retry request: {http_response.status_code} {http_response.content}")
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.error(f"request failed, no retry: {http_response.status_code} {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.failed += 1
                        progress_bar.update()
                    return True

    def wait_execute_retry(self, provider: "OpenAIProvider", progress_bar: ProgressBar | None, use_cache: bool) -> None:
        request_done = False
        while not request_done:
            execute_parallel = False
            with provider.state.wait_semaphore:  # make sure this thread will be next to execute
                while not (request_done or execute_parallel):
                    with provider.state.state_semaphore:
                        state = provider.state.state
                        if self.request["model"] not in state.keys():
                            state[self.request["model"]] = OpenAIRateLimitBudget()

                        state[self.request["model"]] = state[self.request["model"]].consider_time()

                        if state[self.request["model"]].is_enough_for_request(self):
                            state[self.request["model"]] = state[self.request["model"]].decrease_by_request(self)

                            match state[self.request["model"]].mode:
                                case "sequential":  # execute inside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "S"
                                        progress_bar.update_postfix()
                                    if self.execute(provider, progress_bar, use_cache, True):
                                        request_done = True
                                case "parallel":  # execute outside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "P"
                                        progress_bar.update_postfix()
                                    execute_parallel = True
                                case _ as other:
                                    typing.assert_never(other)
                        else:
                            if progress_bar is not None:
                                progress_bar.bottleneck = "L"
                                progress_bar.update_postfix()
                            time.sleep(0.05)  # sleep to wait for rate limit budget

            if execute_parallel:
                if self.execute(provider, progress_bar, use_cache, False):
                    request_done = True


class OpenAIProvider(BaseProvider):
    name: str = "openai"
    additional_tokens: int = 10  # number of additional tokens per message

    def __init__(self) -> None:
        super().__init__()
        self.state.state["total_cost"] = 0.0
        self.state.state["total_calls"] = 0

    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            use_cache: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        pairs = [OpenAIPair(request) for request in requests]

        if progress_bar is not None:
            progress_bar.next_step("load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair, "execute-request", use_cache):
                pairs_to_execute.append(pair)
            else:
                if progress_bar is not None:
                    progress_bar.cached += 1
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            assert "OPENAI_API_KEY" in os.environ.keys(), "missing `OPENAI_API_KEY` in environment variables"

            # determine maximum cost
            reqs_to_execute = [pair.request for pair in pairs_to_execute]
            est_input_tokenss = self.count_tokens(reqs_to_execute, silent, progress_bar)
            for pair, est_input_tokens in zip(pairs_to_execute, est_input_tokenss):
                pair.set_cost_and_usage(est_input_tokens)
            est_max_total_cost = sum(pair.est_max_cost for pair in pairs_to_execute)
            self.user_confirms_cost(est_max_total_cost, budget, silent, progress_bar)

            # sort request to execute longest first, but put one short request first to quickly obtain HTTP header
            pairs_to_execute.sort(key=lambda p: p.est_max_usage, reverse=True)
            pairs_to_execute = pairs_to_execute[-1:] + pairs_to_execute[:-1]

            # execute requests
            if progress_bar is not None:
                progress_bar.next_step("execute requests", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXECUTION_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar, use_cache) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].wait_execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].wait_execute_retry(self, progress_bar, use_cache)

        return [pair.response for pair in pairs]

    def count_tokens(
            self,
            requests: list[Request],
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(requests))

        model = self.check_determine_model(requests)

        import tiktoken
        if "openai_tokenizer_encoding_name" in MODELS[model].keys():
            encoding = tiktoken.get_encoding(MODELS[model]["openai_tokenizer_encoding_name"])
        else:
            encoding = tiktoken.encoding_for_model(model)

        contents, lefts = [], []
        for req in requests:
            lefts.append(len(contents))
            contents += [m["content"] for m in req["messages"]]
        lefts.append(len(contents))

        lengths = [len(tokens) + self.additional_tokens for tokens in
                   encoding.encode_batch(contents, num_threads=NUM_TIKTOKEN_PROCESSES)]
        num_tokens = [sum(lengths[l:r]) for l, r in itertools.pairwise(lefts)]

        if progress_bar is not None:
            progress_bar.update(len(requests))
        return num_tokens

    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(strings))

        import tiktoken
        if "openai_tokenizer_encoding_name" in MODELS[model].keys():
            encoding = tiktoken.get_encoding(MODELS[model]["openai_tokenizer_encoding_name"])
        else:
            encoding = tiktoken.encoding_for_model(model)
        num_tokens = [len(tokens) for tokens in encoding.encode_batch(strings, num_threads=NUM_TIKTOKEN_PROCESSES)]
        if progress_bar is not None:
            progress_bar.update(len(strings))
        return num_tokens

    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        if progress_bar is not None:
            progress_bar.next_step("compute cost", len(responses))

        model = self.check_determine_model(responses)

        costs = []
        for response in responses:
            mode = "default"
            if "service_tier" in response.keys():
                mode = response["service_tier"]
            usd_per_1m_input_tokens = MODELS[model]["usd_per_1m_input_tokens"][mode]
            usd_per_1m_cached_input_tokens = MODELS[model]["usd_per_1m_cached_input_tokens"][mode]
            usd_per_1m_output_tokens = MODELS[model]["usd_per_1m_output_tokens"][mode]

            # TODO: implement OpenAI responses API as its own provider
            api = MODELS[model].get("model_api","completion")
            prompt_tokens_key = "prompt_tokens" if api == "completion" else "input_tokens"
            output_tokens_key = "completion_tokens" if api == "completion" else "output_tokens"

            cost_ = 0
            if ignore_token_caching:
                cost_ += response["usage"][prompt_tokens_key] * usd_per_1m_input_tokens / 1_000_000
            else:
                cached_tokens = response["usage"][f"{prompt_tokens_key}_details"]["cached_tokens"]
                cost_ += cached_tokens * usd_per_1m_cached_input_tokens / 1_000_000
                prompt_tokens = response["usage"][prompt_tokens_key] - cached_tokens
                cost_ += prompt_tokens * usd_per_1m_input_tokens / 1_000_000
            cost_ += response["usage"][output_tokens_key] * usd_per_1m_output_tokens / 1_000_000
            costs.append(cost_)
            if progress_bar is not None:
                progress_bar.update()
        return costs

    def transform_request_from_openai_to_self(self, request: Request) -> Request:
        return request

    def transform_request_from_ollama_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_request_from_vllm_to_self(self, request: Request) -> Request:
        return request  # vLLM-API should be OpenAI-compatible

    def transform_request_from_anthropic_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_response_from_self_to_openai(self, response: Response) -> Response:
        return response

    def transform_response_from_self_to_ollama(self, response: Response) -> Response:
        res = self.autodict()
        for r_key, r_value in response.items():
            match r_key:
                case "model":
                    res["model"] = r_value
                case "choices":
                    assert len(r_value) == 1, \
                        "transform response from openai to ollama: cannot handle multiple `choices` list with multiple entries"
                    for c_key, c_value in r_value[0].items():
                        match c_key:
                            case "message":
                                for m_key, m_value in c_value.items():
                                    match m_key:
                                        case "role":
                                            res["message"]["role"] = c_value
                                        case "content":
                                            res["message"]["content"] = c_value
                                        case _ as m_key:
                                            self.trans_warning(
                                                f"transform response from openai to ollama: ignored unknown value `response[\"choices\"][0][\"message\"][\"{m_key}\"] = {m_value}`"
                                            )
                            case "finish_reason":
                                match c_value:
                                    case "length":
                                        res["done_reason"] = "length"
                                    case "stop":
                                        res["done_reason"] = "stop"
                                    case _ as c_value:
                                        self.trans_warning(
                                            f"transform response from openai to ollama: ignored unknown value `response[\"choices\"][0][\"finish_reason\"] = {c_value}`"
                                        )
                            case _ as c_key:
                                self.trans_warning(
                                    f"transform response from openai to ollama: ignored unknown key `response[\"choices\"][0][\"{c_key}\"] = {c_value}`"
                                )
                case "usage":
                    for u_key, u_value in r_value.items():
                        match u_key:
                            case "prompt_tokens":
                                res["prompt_eval_count"] = u_value
                            case "completion_tokens":
                                res["eval_count"] = u_value
                            case _ as u_key:
                                self.trans_warning(
                                    f"transform response from openai to ollama: ignored unknown key-value `response[\"usage\"][\"{u_key}\"] = {u_value}`"
                                )
                case _ as r_key:
                    self.trans_warning(
                        f"transform response from openai to ollama: ignored unknown key-value `response[\"{r_key}\"] = {r_value}`"
                    )
        return self.autodict_to_dict(res)

    def transform_response_from_self_to_vllm(self, response: Response) -> Response:
        return response  # vLLM-API should be OpenAI-compatible

    def transform_response_from_self_to_anthropic(self, response: Response) -> Response:
        raise NotImplementedError()


PROVIDERS[OpenAIProvider.name] = OpenAIProvider()


########################################################################################################################
# Ollama implementation
########################################################################################################################


@dataclasses.dataclass
class OllamaPair(BasePair):
    num_tries: int = 0

    def execute(self, provider: "OllamaProvider", progress_bar: ProgressBar | None, use_cache: bool) -> bool:
        http_response = requests.post(
            url="http://localhost:11434/api/chat",
            json=self.request
        )
        self.num_tries += 1

        with provider.state.state_semaphore:
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self, "execute-request", use_cache)
                    if progress_bar is not None:
                        progress_bar.update()
                    return True
                case status_code if status_code in RETRY_ON_HTTP_STATUS_CODES and self.num_tries < MAX_TRIES:
                    logger.info(f"retry request: {http_response.status_code} {http_response.content}")
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.error(f"request failed, no retry: {http_response.status_code} {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.failed += 1
                        progress_bar.update()
                    return True

    def execute_retry(self, provider: "OllamaProvider", progress_bar: ProgressBar | None, use_cache: bool) -> None:
        while not self.execute(provider, progress_bar, use_cache):
            pass


class OllamaProvider(BaseProvider):
    name: str = "ollama"
    additional_tokens: int = 10  # number of additional tokens per message
    hf_tokenizer_cache: dict  # type dict[str, tokenizers.Tokenizer], avoid unnecessary import

    def __init__(self) -> None:
        super().__init__()
        self.hf_tokenizer_cache = {}

    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            use_cache: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        pairs = [OllamaPair(request) for request in requests]

        for pair in pairs:
            assert "stream" in pair.request.keys() and not pair.request["stream"], \
                "execution requires `request[\"stream\"] = False`"

            if progress_bar is not None:
                progress_bar.next_step("load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair, "execute-request", use_cache):
                pairs_to_execute.append(pair)
            else:
                if progress_bar is not None:
                    progress_bar.cached += 1
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            # execute requests
            if progress_bar is not None:
                progress_bar.next_step("execute requests", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXECUTION_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar, use_cache) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].execute_retry(self, progress_bar, use_cache)

        return [pair.response for pair in pairs]

    def count_tokens(
            self,
            requests: list[Request],
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(requests))

        model = self.check_determine_model(requests)

        import tokenizers
        assert "HF_TOKEN" in os.environ.keys(), "missing `HF_TOKEN` in environment variables"
        hf_model_name = MODELS[model]["hf_model_name"]

        if hf_model_name in self.hf_tokenizer_cache.keys():
            tokenizer = self.hf_tokenizer_cache[hf_model_name]
        else:
            tokenizer = tokenizers.Tokenizer.from_pretrained(hf_model_name)
            self.hf_tokenizer_cache[hf_model_name] = tokenizer

        contents, lefts = [], []
        for req in requests:
            lefts.append(len(contents))
            contents += [m["content"] for m in req["messages"]]
        lefts.append(len(contents))

        lengths = [len(e.tokens) + self.additional_tokens for e in
                   tokenizer.encode_batch_fast(contents, add_special_tokens=False)]
        num_tokens = [sum(lengths[l:r]) for l, r in itertools.pairwise(lefts)]

        if progress_bar is not None:
            progress_bar.update(len(requests))
        return num_tokens

    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(strings))

        import tokenizers
        assert "HF_TOKEN" in os.environ.keys(), "missing `HF_TOKEN` in environment variables"
        hf_model_name = MODELS[model]["hf_model_name"]

        if hf_model_name in self.hf_tokenizer_cache.keys():
            tokenizer = self.hf_tokenizer_cache[hf_model_name]
        else:
            tokenizer = tokenizers.Tokenizer.from_pretrained(hf_model_name)
            self.hf_tokenizer_cache[hf_model_name] = tokenizer

        num_tokens = [len(e.tokens) for e in tokenizer.encode_batch_fast(strings, add_special_tokens=False)]

        if progress_bar is not None:
            progress_bar.update(len(strings))
        return num_tokens

    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        if progress_bar is not None:
            progress_bar.next_step("compute cost", len(responses))

        model = self.check_determine_model(responses)

        mode = "default"
        usd_per_1m_input_tokens = MODELS[model]["usd_per_1m_input_tokens"][mode]
        usd_per_1m_cached_input_tokens = MODELS[model]["usd_per_1m_cached_input_tokens"][mode]
        usd_per_1m_output_tokens = MODELS[model]["usd_per_1m_output_tokens"][mode]

        costs = []
        for response in responses:
            cost_ = response["prompt_eval_count"] * usd_per_1m_input_tokens / 1_000_000 \
                    + response["eval_count"] * usd_per_1m_output_tokens / 1_000_000
            costs.append(cost_)
            if progress_bar is not None:
                progress_bar.update()

        return costs

    def transform_request_from_openai_to_self(self, request: Request) -> Request:
        res = self.autodict()
        for r_key, r_value in request.items():
            match r_key:
                case "model":
                    res["model"] = r_value
                case "messages":
                    res["messages"] = []
                    for message in r_value:
                        mes = self.autodict()
                        for m_key, m_value in message.items():
                            match m_key:
                                case "role":
                                    mes["role"] = m_value
                                case "content":
                                    mes["content"] = m_value
                                case _ as m_key:
                                    self.trans_warning(
                                        f"transform request from openai to ollama: ignored unknown key `request[\"messages\"][i][\"{m_key}\"] = {m_value}`"
                                    )
                        res["messages"].append(mes)
                case "seed":
                    res["options"]["seed"] = r_value
                case "temperature":
                    res["options"]["temperature"] = r_value
                case "max_tokens" | "max_completion_tokens":
                    res["options"]["num_predict"] = r_value
                case _ as r_key:
                    self.trans_warning(
                        f"transform request from openai to ollama: ignored unknown key `request[\"{r_key}\"]= {r_value}`"
                    )

        res["stream"] = False
        res["options"]["num_ctx"] = MODELS[request["model"]]["max_context"]
        return self.autodict_to_dict(res)

    def transform_request_from_ollama_to_self(self, request: Request) -> Request:
        return request

    def transform_request_from_vllm_to_self(self, request: Request) -> Request:
        return self.transform_request_from_openai_to_self(request)  # vLLM-API should be OpenAI-compatible

    def transform_request_from_anthropic_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_response_from_self_to_openai(self, response: Response) -> Response:
        res = self.autodict()
        for r_key, r_value in response.items():
            match r_key:
                case "model":
                    res["model"] = r_value
                case "message":
                    if "choices" not in res.keys():
                        res["choices"] = [self.autodict()]
                    for m_key, m_value in r_value.items():
                        match m_key:
                            case "role":
                                res["choices"][0]["message"]["role"] = m_value
                            case "content":
                                res["choices"][0]["message"]["content"] = m_value
                            case _ as m_key:
                                self.trans_warning(
                                    f"transform response from ollama to openai: ignored unknown key `response[\"message\"][\"{m_key}\"] = {m_value}`"
                                )
                case "prompt_eval_count":
                    res["usage"]["prompt_tokens"] = r_value
                case "eval_count":
                    res["usage"]["completion_tokens"] = r_value
                case "done_reason":
                    if "choices" not in res.keys():
                        res["choices"] = [self.autodict()]
                    match r_value:
                        case "length":
                            res["choices"][0]["finish_reason"] = "length"
                        case "stop":
                            res["choices"][0]["finish_reason"] = "stop"
                        case _:
                            self.trans_warning(
                                f"transform response from ollama to openai: ignored unknown value `response[\"done_reason\"] = {r_value}`"
                            )
                case _ as r_key:
                    self.trans_warning(
                        f"transform response from ollama to openai: ignored unknown key `response[\"{r_key}\"] = {r_value}`"
                    )
        return self.autodict_to_dict(res)

    def transform_response_from_self_to_ollama(self, response: Response) -> Response:
        return response

    def transform_response_from_self_to_vllm(self, response: Response) -> Response:
        return self.transform_response_from_self_to_openai(response)  # vLLM-API should be OpenAI-compatible

    def transform_response_from_self_to_anthropic(self, response: Response) -> Response:
        raise NotImplementedError()


PROVIDERS[OllamaProvider.name] = OllamaProvider()


########################################################################################################################
# vLLM implementation
########################################################################################################################


@dataclasses.dataclass
class VLLMPair(BasePair):
    num_tries: int = 0

    def execute(self, provider: "VLLMProvider", progress_bar: ProgressBar | None, use_cache: bool) -> bool:
        http_response = requests.post(
            url="http://localhost:8000/v1/chat/completions",
            json=self.request
        )
        self.num_tries += 1

        with provider.state.state_semaphore:
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self, "execute-request", use_cache)
                    if progress_bar is not None:
                        progress_bar.update()
                    return True
                case status_code if status_code in RETRY_ON_HTTP_STATUS_CODES and self.num_tries < MAX_TRIES:
                    logger.info(f"retry request: {http_response.status_code} {http_response.content}")
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.error(f"request failed, no retry: {http_response.status_code} {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.failed += 1
                        progress_bar.update()
                    return True

    def execute_retry(self, provider: "VLLMProvider", progress_bar: ProgressBar | None, use_cache: bool) -> None:
        while not self.execute(provider, progress_bar, use_cache):
            pass


class VLLMProvider(BaseProvider):
    name: str = "vllm"
    additional_tokens: int = 10  # number of additional tokens per message
    hf_tokenizer_cache: dict  # type dict[str, tokenizers.Tokenizer], avoid unnecessary import

    def __init__(self) -> None:
        super().__init__()
        self.hf_tokenizer_cache = {}

    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            use_cache: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        pairs = [VLLMPair(request) for request in requests]

        if progress_bar is not None:
            progress_bar.next_step("load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair, "execute-request", use_cache):
                pairs_to_execute.append(pair)
            else:
                if progress_bar is not None:
                    progress_bar.cached += 1
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            # execute requests
            if progress_bar is not None:
                progress_bar.next_step("execute requests", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXECUTION_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar, use_cache) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].execute_retry(self, progress_bar, use_cache)

        return [pair.response for pair in pairs]

    def count_tokens(
            self,
            requests: list[Request],
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(requests))

        model = self.check_determine_model(requests)

        import tokenizers
        assert "HF_TOKEN" in os.environ.keys(), "missing `HF_TOKEN` in environment variables"
        hf_model_name = MODELS[model]["hf_model_name"]

        if hf_model_name in self.hf_tokenizer_cache.keys():
            tokenizer = self.hf_tokenizer_cache[hf_model_name]
        else:
            tokenizer = tokenizers.Tokenizer.from_pretrained(hf_model_name)
            self.hf_tokenizer_cache[hf_model_name] = tokenizer

        contents, lefts = [], []
        for req in requests:
            lefts.append(len(contents))
            contents += [m["content"] for m in req["messages"]]
        lefts.append(len(contents))

        lengths = [len(e.tokens) + self.additional_tokens for e in
                   tokenizer.encode_batch_fast(contents, add_special_tokens=False)]
        num_tokens = [sum(lengths[l:r]) for l, r in itertools.pairwise(lefts)]

        if progress_bar is not None:
            progress_bar.update(len(requests))
        return num_tokens

    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(strings))

        import tokenizers
        assert "HF_TOKEN" in os.environ.keys(), "missing `HF_TOKEN` in environment variables"
        hf_model_name = MODELS[model]["hf_model_name"]

        if hf_model_name in self.hf_tokenizer_cache.keys():
            tokenizer = self.hf_tokenizer_cache[hf_model_name]
        else:
            tokenizer = tokenizers.Tokenizer.from_pretrained(hf_model_name)
            self.hf_tokenizer_cache[hf_model_name] = tokenizer

        num_tokens = [len(e.tokens) for e in tokenizer.encode_batch_fast(strings, add_special_tokens=False)]

        if progress_bar is not None:
            progress_bar.update(len(strings))
        return num_tokens

    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        raise NotImplementedError()

    def transform_request_from_openai_to_self(self, request: Request) -> Request:
        return request  # vLLM-API should be OpenAI-compatible

    def transform_request_from_ollama_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_request_from_vllm_to_self(self, request: Request) -> Request:
        return request

    def transform_request_from_anthropic_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_response_from_self_to_openai(self, response: Response) -> Response:
        return response  # vLLM-API should be OpenAI-compatible

    def transform_response_from_self_to_ollama(self, response: Response) -> Response:
        raise NotImplementedError()

    def transform_response_from_self_to_vllm(self, response: Response) -> Response:
        return response

    def transform_response_from_self_to_anthropic(self, response: Response) -> Response:
        raise NotImplementedError()


PROVIDERS[VLLMProvider.name] = VLLMProvider()


########################################################################################################################
# Anthropic implementation
########################################################################################################################


@dataclasses.dataclass
class AnthropicCountTokensRateLimitBudget:
    mode: typing.Literal["sequential", "parallel"] = "sequential"
    rpm: float | None = None
    r: float | None = None
    last_update: float = dataclasses.field(default_factory=time.time)

    def is_enough_for_request(self, pair: "AnthropicCountTokensPair") -> bool:
        return self.r is None or self.r >= 1

    def consider_time(self) -> typing.Self:
        now = time.time()
        delta = now - self.last_update
        if self.rpm is not None and self.r is not None:
            self.r = min(self.rpm, self.r + self.rpm * delta / 60)
        self.last_update = now
        return self

    def decrease_by_request(self, pair: "AnthropicPair") -> typing.Self:
        if self.r is not None:
            self.r -= 1
        return self

    def set_from_headers(self, headers: dict[str, typing.Any]) -> typing.Self:
        if "anthropic-ratelimit-requests-limit" in headers.keys():
            self.rpm = float(headers["anthropic-ratelimit-requests-limit"])
        if "anthropic-ratelimit-requests-remaining" in headers.keys():
            header_r = float(headers["anthropic-ratelimit-requests-remaining"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        return self

    def to_par(self) -> typing.Self:
        self.mode = "parallel"
        return self

    def to_seq(self) -> typing.Self:
        self.mode = "sequential"
        return self


@dataclasses.dataclass
class AnthropicRateLimitBudget:
    mode: typing.Literal["sequential", "parallel"] = "sequential"
    rpm: float | None = None
    tpm: float | None = None
    itpm: float | None = None
    otpm: float | None = None
    r: float | None = None
    t: float | None = None
    it: float | None = None
    ot: float | None = None
    last_update: float = dataclasses.field(default_factory=time.time)

    def is_enough_for_request(self, pair: "AnthropicPair") -> bool:
        return (self.r is None or self.r >= 1) \
            and (self.t is None or self.t >= pair.est_max_usage) \
            and (self.it is None or self.it >= pair.est_max_input_usage) \
            and (self.ot is None or self.ot >= pair.est_max_output_usage)

    def consider_time(self) -> typing.Self:
        now = time.time()
        delta = now - self.last_update
        if self.rpm is not None and self.r is not None:
            self.r = min(self.rpm, self.r + self.rpm * delta / 60)
        if self.tpm is not None and self.t is not None:
            self.t = min(self.tpm, self.t + self.tpm * delta / 60)
        if self.itpm is not None and self.it is not None:
            self.it = min(self.itpm, self.it + self.itpm * delta / 60)
        if self.otpm is not None and self.ot is not None:
            self.ot = min(self.otpm, self.ot + self.otpm * delta / 60)
        self.last_update = now
        return self

    def decrease_by_request(self, pair: "AnthropicPair") -> typing.Self:
        if self.r is not None:
            self.r -= 1
        if self.t is not None:
            self.t -= pair.est_max_usage
        if self.it is not None:
            self.it -= pair.est_max_input_usage
        if self.ot is not None:
            self.ot -= pair.est_max_output_usage
        return self

    def set_from_headers(self, headers: dict[str, typing.Any]) -> typing.Self:
        if "anthropic-ratelimit-requests-limit" in headers.keys():
            self.rpm = float(headers["anthropic-ratelimit-requests-limit"])
        if "anthropic-ratelimit-tokens-limit" in headers.keys():
            self.tpm = float(headers["anthropic-ratelimit-tokens-limit"])
        if "anthropic-ratelimit-input-tokens-limit" in headers.keys():
            self.itpm = float(headers["anthropic-ratelimit-input-tokens-limit"])
        if "anthropic-ratelimit-output-tokens-limit" in headers.keys():
            self.otpm = float(headers["anthropic-ratelimit-output-tokens-limit"])
        if "anthropic-ratelimit-requests-remaining" in headers.keys():
            header_r = float(headers["anthropic-ratelimit-requests-remaining"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        if "anthropic-ratelimit-tokens-remaining" in headers.keys():
            header_t = float(headers["anthropic-ratelimit-tokens-remaining"])
            if self.t is None or self.t > header_t:
                self.t = header_t
        if "anthropic-ratelimit-input-tokens-remaining" in headers.keys():
            header_it = float(headers["anthropic-ratelimit-input-tokens-remaining"])
            if self.it is None or self.it > header_it:
                self.it = header_it
        if "anthropic-ratelimit-output-tokens-remaining" in headers.keys():
            header_ot = float(headers["anthropic-ratelimit-output-tokens-remaining"])
            if self.ot is None or self.ot > header_ot:
                self.ot = header_ot
        return self

    def to_par(self) -> typing.Self:
        self.mode = "parallel"
        return self

    def to_seq(self) -> typing.Self:
        self.mode = "sequential"
        return self


@dataclasses.dataclass
class AnthropicCountTokensPair(BasePair):

    def execute(
            self,
            provider: "AnthropicProvider",
            progress_bar: ProgressBar | None,
            use_cache: bool,
            has_semaphore: bool
    ) -> bool:
        # TODO: check that these still are all the relevant parts of the request
        req = {k: v for k, v in self.request.items() if
               k in {"messages", "model", "system", "tool_choice", "tools", "thinking"}}

        http_response = requests.post(
            url="https://api.anthropic.com/v1/messages/count_tokens",
            json=req,
            headers={
                "content-type": "application/json",
                "x-api-key": f"{os.environ['ANTHROPIC_API_KEY']}",
                "anthropic-version": "2023-06-01"
            }
        )
        self.num_tries += 1

        with provider.state.state_semaphore if not has_semaphore else contextlib.nullcontext():
            provider.state.state["total_count_tokens_calls"] = provider.state.state["total_count_tokens_calls"] + 1
            provider.state.state[f"{self.request["model"]}-count-tokens"] = provider.state.state[
                f"{self.request["model"]}-count-tokens"].set_from_headers(
                http_response.headers
            )
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self, "count-tokens", use_cache)
                    provider.state.state[f"{self.request["model"]}-count-tokens"] = provider.state.state[
                        f"{self.request["model"]}-count-tokens"].to_par()
                    if progress_bar is not None:
                        progress_bar.update()
                    return True
                case 429:
                    logger.info("retry count tokens request due to rate limit error")
                    provider.state.state[f"{self.request["model"]}-count-tokens"] = provider.state.state[
                        f"{self.request["model"]}-count-tokens"].to_seq()
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case status_code if status_code in RETRY_ON_HTTP_STATUS_CODES and self.num_tries < MAX_TRIES:
                    logger.info(f"retry request: {http_response.status_code} {http_response.content}")
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.error(
                        f"count tokens request failed, no retry: {http_response.status_code} {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.update()
                    return True

    def wait_execute_retry(self, provider: "AnthropicProvider", progress_bar: ProgressBar | None,
                           use_cache: bool) -> None:
        request_done = False
        while not request_done:
            execute_parallel = False
            with provider.state.wait_semaphore:  # make sure this thread will be next to execute
                while not (request_done or execute_parallel):
                    with provider.state.state_semaphore:
                        state = provider.state.state
                        if f"{self.request["model"]}-count-tokens" not in state.keys():
                            state[f"{self.request["model"]}-count-tokens"] = AnthropicCountTokensRateLimitBudget()

                        state[f"{self.request["model"]}-count-tokens"] = \
                            state[f"{self.request["model"]}-count-tokens"].consider_time()

                        if state[f"{self.request["model"]}-count-tokens"].is_enough_for_request(self):
                            state[f"{self.request["model"]}-count-tokens"] = \
                                state[f"{self.request["model"]}-count-tokens"].decrease_by_request(self)

                            match state[f"{self.request["model"]}-count-tokens"].mode:
                                case "sequential":  # execute inside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "S"
                                        progress_bar.update_postfix()
                                    if self.execute(provider, progress_bar, use_cache, True):
                                        request_done = True
                                case "parallel":  # execute outside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "P"
                                        progress_bar.update_postfix()
                                    execute_parallel = True
                                case _ as other:
                                    typing.assert_never(other)
                        else:
                            if progress_bar is not None:
                                progress_bar.bottleneck = "L"
                                progress_bar.update_postfix()
                            time.sleep(0.05)  # sleep to wait for rate limit budget

            if execute_parallel:
                if self.execute(provider, progress_bar, use_cache, False):
                    request_done = True


@dataclasses.dataclass
class AnthropicPair(BasePair):
    est_max_cost: float | None = None
    est_max_usage: int | None = None
    est_max_input_usage: int | None = None
    est_max_output_usage: int | None = None

    def set_cost_and_usage(self, est_input_tokens: int) -> None:
        model_params = MODELS[self.request["model"]]

        self.est_max_input_usage = est_input_tokens
        self.est_max_output_usage = self.request["max_tokens"]
        self.est_max_usage = self.est_max_input_usage + self.est_max_output_usage

        # TODO: take prompt length, caching, and caching duration into account
        self.est_max_cost = self.est_max_input_usage * model_params["usd_per_1m_input_tokens"]["default"] / 1_000_000 \
                            + self.est_max_output_usage * model_params["usd_per_1m_output_tokens"][
                                "default"] / 1_000_000

    def execute(
            self,
            provider: "AnthropicProvider",
            progress_bar: ProgressBar | None,
            use_cache: bool,
            has_semaphore: bool
    ) -> bool:
        http_response = requests.post(
            url="https://api.anthropic.com/v1/messages",
            json=self.request,
            headers={
                "content-type": "application/json",
                "x-api-key": f"{os.environ['ANTHROPIC_API_KEY']}",
                "anthropic-version": "2023-06-01"
            }
        )
        self.num_tries += 1

        with provider.state.state_semaphore if not has_semaphore else contextlib.nullcontext():
            provider.state.state["total_calls"] = provider.state.state["total_calls"] + 1
            provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].set_from_headers(
                http_response.headers
            )
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self, "execute-request", use_cache)
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_par()
                    cost_ = provider.cost([self.response], False, True, None)[0]
                    provider.state.state["total_cost"] = provider.state.state["total_cost"] + cost_ - self.est_max_cost
                    if progress_bar is not None:
                        progress_bar.cost += cost_
                        progress_bar.update()
                    return True
                case 429:
                    logger.info("retry request due to rate limit error")
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_seq()
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case status_code if status_code in RETRY_ON_HTTP_STATUS_CODES and self.num_tries < MAX_TRIES:
                    logger.info(f"retry request: {http_response.status_code} {http_response.content}")
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.error(f"request failed, no retry: {http_response.status_code} {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.failed += 1
                        progress_bar.update()
                    return True

    def wait_execute_retry(self, provider: "AnthropicProvider", progress_bar: ProgressBar | None,
                           use_cache: bool) -> None:
        request_done = False
        while not request_done:
            execute_parallel = False
            with provider.state.wait_semaphore:  # make sure this thread will be next to execute
                while not (request_done or execute_parallel):
                    with provider.state.state_semaphore:
                        state = provider.state.state
                        if self.request["model"] not in state.keys():
                            state[self.request["model"]] = AnthropicRateLimitBudget()

                        state[self.request["model"]] = state[self.request["model"]].consider_time()

                        if state[self.request["model"]].is_enough_for_request(self):
                            state[self.request["model"]] = state[self.request["model"]].decrease_by_request(self)

                            match state[self.request["model"]].mode:
                                case "sequential":  # execute inside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "S"
                                        progress_bar.update_postfix()
                                    if self.execute(provider, progress_bar, use_cache, True):
                                        request_done = True
                                case "parallel":  # execute outside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "P"
                                        progress_bar.update_postfix()
                                    execute_parallel = True
                                case _ as other:
                                    typing.assert_never(other)
                        else:
                            if progress_bar is not None:
                                progress_bar.bottleneck = "L"
                                progress_bar.update_postfix()
                            time.sleep(0.05)  # sleep to wait for rate limit budget

            if execute_parallel:
                if self.execute(provider, progress_bar, use_cache, False):
                    request_done = True


class AnthropicProvider(BaseProvider):
    name: str = "anthropic"
    additional_tokens: int = 7  # number of additional tokens when token counting for strings

    def __init__(self) -> None:
        super().__init__()
        self.state.state["total_cost"] = 0.0
        self.state.state["total_calls"] = 0
        self.state.state["total_count_tokens_calls"] = 0

    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            use_cache: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        pairs = [AnthropicPair(request) for request in requests]

        if progress_bar is not None:
            progress_bar.next_step("load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair, "execute-request", use_cache):
                pairs_to_execute.append(pair)
            else:
                if progress_bar is not None:
                    progress_bar.cached += 1
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            assert "ANTHROPIC_API_KEY" in os.environ.keys(), "missing `ANTHROPIC_API_KEY` in environment variables"

            # determine maximum cost
            reqs_to_execute = [pair.request for pair in pairs_to_execute]
            est_input_tokenss = self.count_tokens(reqs_to_execute, silent, progress_bar)
            for pair, est_input_tokens in zip(pairs_to_execute, est_input_tokenss):
                pair.set_cost_and_usage(est_input_tokens)
            est_max_total_cost = sum(pair.est_max_cost for pair in pairs_to_execute)
            self.user_confirms_cost(est_max_total_cost, budget, silent, progress_bar)

            # sort request to execute longest first, but put one short request first to quickly obtain HTTP header
            pairs_to_execute.sort(key=lambda p: p.est_max_usage, reverse=True)
            pairs_to_execute = pairs_to_execute[-1:] + pairs_to_execute[:-1]

            # execute requests
            if progress_bar is not None:
                progress_bar.next_step("execute requests", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXECUTION_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar, use_cache) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].wait_execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].wait_execute_retry(self, progress_bar, use_cache)

        return [pair.response for pair in pairs]

    def count_tokens(
            self,
            requests: list[Request],
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        pairs = [AnthropicCountTokensPair(request) for request in requests]

        if progress_bar is not None:
            progress_bar.next_step("count tokens - load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair, "count-tokens", True):  # TODO: make configurable
                pairs_to_execute.append(pair)
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            assert "ANTHROPIC_API_KEY" in os.environ.keys(), "missing `ANTHROPIC_API_KEY` in environment variables"

            # execute count tokens requests
            if progress_bar is not None:
                progress_bar.next_step("count tokens", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXECUTION_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar, True) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].wait_execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].wait_execute_retry(self, progress_bar, True)

        num_tokens = [pair.response["input_tokens"] for pair in pairs]

        if progress_bar is not None:
            progress_bar.update(len(requests))
        return num_tokens

    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        requests_ = [{"model": model, "messages": [{"role": "user", "content": s}]} for s in strings]
        num_tokens = self.count_tokens(requests_, silent, progress_bar)
        num_tokens = [nt - self.additional_tokens for nt in num_tokens]
        return num_tokens

    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        if progress_bar is not None:
            progress_bar.next_step("compute cost", len(responses))

        model = self.check_determine_model(responses)

        costs = []
        for response in responses:
            cost_ = 0
            # TODO: take prompt length and caching duration into account
            if response["usage"]["cache_creation_input_tokens"] is not None:
                if not ignore_token_caching:
                    cost_ += response["usage"]["cache_creation_input_tokens"] \
                             * MODELS[model]["usd_per_1m_input_tokens"]["default_cache_write"] / 1_000_000
                else:
                    cost_ += response["usage"]["cache_creation_input_tokens"] \
                             * MODELS[model]["usd_per_1m_input_tokens"]["default"] / 1_000_000

            if response["usage"]["cache_read_input_tokens"] is not None:
                if not ignore_token_caching:
                    cost_ += response["usage"]["cache_read_input_tokens"] \
                             * MODELS[model]["usd_per_1m_cached_input_tokens"]["default"] / 1_000_000
                else:
                    cost_ += response["usage"]["cache_read_input_tokens"] \
                             * MODELS[model]["usd_per_1m_input_tokens"]["default"] / 1_000_000

            if response["usage"]["input_tokens"] is not None:
                cost_ += response["usage"]["input_tokens"] \
                         * MODELS[model]["usd_per_1m_input_tokens"]["default"] / 1_000_000

            if response["usage"]["output_tokens"] is not None:
                cost_ += response["usage"]["output_tokens"] \
                         * MODELS[model]["usd_per_1m_output_tokens"]["default"] / 1_000_000

            costs.append(cost_)
            if progress_bar is not None:
                progress_bar.update()
        return costs

    def transform_request_from_openai_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_request_from_ollama_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_request_from_vllm_to_self(self, request: Request) -> Request:
        raise NotImplementedError()

    def transform_request_from_anthropic_to_self(self, request: Request) -> Request:
        return request

    def transform_response_from_self_to_openai(self, response: Response) -> Response:
        raise NotImplementedError()

    def transform_response_from_self_to_ollama(self, response: Response) -> Response:
        raise NotImplementedError()

    def transform_response_from_self_to_vllm(self, response: Response) -> Response:
        raise NotImplementedError()

    def transform_response_from_self_to_anthropic(self, response: Response) -> Response:
        return response


PROVIDERS[AnthropicProvider.name] = AnthropicProvider()