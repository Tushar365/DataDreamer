import gc
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, cast

import unify  
import tiktoken
from datasets.fingerprint import Hasher
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)
from tiktoken import Encoding

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


@lru_cache(maxsize=None)
def _normalize_model_name(model_name: str) -> str:
    # Adapt this to Unify's model naming conventions
    return model_name


class UnifyException(Exception):
    pass


class UnifyAI(LLM):
    def __init__(
        self,
        model_name: str,
        system_prompt: None | str = None,
        organization: None | str = None,
        api_key: None | str = None,
        base_url: None | str = None,
        api_version: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kwargs = kwargs
        self.system_prompt = system_prompt

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

    @cached_property
    def retry_wrapper(self):
        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(unify.RateLimitError),  # Replace with Unify's RateLimitError
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        # Add more retry decorators for other Unify exceptions
        def _retry_wrapper(func, **kwargs):
            return func(**kwargs)

        _retry_wrapper.__wrapped__.__module__ = None  # type: ignore[attr-defined]
        _retry_wrapper.__wrapped__.__qualname__ = f"{self.__class__.__name__}.run"  # type: ignore[attr-defined]
        return _retry_wrapper

    @cached_property
    def clients(self) -> unify.clients:  
        # Adapt this to Unify's client initialization
        return unify.clients(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.kwargs,
        )

    @cached_property
    def tokenizer(self) -> Encoding:
        # Adapt this to Unify's tokenizer
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:
        # Adapt this to Unify's context length limits
        return 4096 - max_new_tokens

    def _get_max_output_length(self) -> None | int:
        # Adapt this to Unify's output length limits
        return None

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        return len(self.tokenizer.encode(value))

    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: None | int = None,
        **kwargs,
    ) -> list[str] | list[list[str]]:
        prompts = inputs

        # Check max_new_tokens
        max_new_tokens = _check_max_new_tokens_possible(
            self=self,
            max_length_func=max_length_func,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        # Set temperature and top_p
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature, top_p=top_p
        )

        # Run the model using Unify's API
        def get_generated_texts(self, kwargs, prompt) -> list[str]:
            # Adapt this to Unify's text generation API
            response = self.retry_wrapper(
                func=self.client.generate_text,  # Replace with Unify's text generation method
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
                **kwargs,
            )
            return [response]  # Adapt this based on Unify's response format

        if batch_size not in self.executor_pools:
            self.executor_pools[batch_size] = ThreadPoolExecutor(max_workers=batch_size)
        generated_texts_batch = list(
            self.executor_pools[batch_size].map(
                partial(get_generated_texts, self, kwargs), prompts
            )
        )

        if n == 1:
            return [batch[0] for batch in generated_texts_batch]
        else:
            return generated_texts_batch

    # ... (rest of the code remains similar to the OpenAI class, but adapted for Unify)

    def unload_model(self):
        # Delete cached client and tokenizer
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached client or tokenizer before serializing
        state.pop("retry_wrapper", None)
        state.pop("client", None)
        state.pop("tokenizer", None)
        state["executor_pools"].clear()

        return state


__all__ = ["UnifyAI"]
