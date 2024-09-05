import gc
import logging
import sys
import openai
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, cast,Optional

from datasets.fingerprint import Hasher
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)

from unify.clients import Unify, AsyncUnify
from openai.types.chat import ChatCompletion  

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


class UnifyException(Exception):
    pass


class UnifyAI(LLM):
    """
    A class for interacting with the Unify.ai platform for language model interactions.

    This class supports both synchronous and asynchronous operations through separate
    Unify clients. 
    """
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
        provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the UnifyAI instance.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): A system prompt to provide context to the model.
            organization (str, optional): The organization associated with the API key.
            api_key (str, optional): The API key for accessing the Unify platform.
            base_url (str, optional): The base URL for the Unify API.
            api_version (str, optional): The version of the Unify API to use.
            retry_on_fail (bool, default=True): Whether to retry API calls on failure.
            cache_folder_path (str, optional): The path to the cache folder.
            provider (str, optional): The name of the language model provider (e.g., "openai").
            **kwargs: Additional keyword arguments to pass to the Unify clients.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kwargs = kwargs
        self.system_prompt = system_prompt
        self.provider = provider

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

        # Initialize the Unify clients
        self._client = self._get_client()
        self._async_client = self._get_async_client()

    @cached_property
    def retry_wrapper(self):
        """
        Creates a retry wrapper function using `tenacity` to handle API call failures.

        Retries on `unify.RateLimitError` and potentially other Unify exceptions.

        Returns:
            Callable: The retry wrapper function.
        """
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        def _retry_wrapper(func, **kwargs):
            return func(**kwargs)

        return retry(
            retry=retry_if_exception_type(unify.RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),
            reraise=True,
        )(_retry_wrapper)

    def _get_client(self) -> UnifyClient:
        """
        Initializes and returns a synchronous Unify client.

        Returns:
            UnifyClient: The synchronous Unify client.
        """
        try:
            return UnifyClient(
                api_key=self.api_key,
                endpoint=f"{self.model_name}@{self.provider}",
                **self.kwargs,
            )
        except Exception as e:
            raise UnifyException(f"Failed to initialize Unify client: {str(e)}")

    def _get_async_client(self) -> AsyncUnify:
        """
        Initializes and returns an asynchronous Unify client.

        Returns:
            AsyncUnify: The asynchronous Unify client.
        """
        try:
            return AsyncUnify(
                api_key=self.api_key,
                endpoint=f"{self.model_name}@{self.provider}",
                **self.kwargs,
            )
        except Exception as e:
            raise UnifyException(f"Failed to initialize Async Unify client: {str(e)}")

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        # TODO:  Implement proper token counting for Unify.
        # For now, we are using a simple placeholder. 
        return len(value.split()) 

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
            # TODO:  Adapt to Unify's text generation API - refer to documentation
            # Below is a placeholder - ensure parameter names and response handling are correct
            response = self.retry_wrapper(
                func=self.client.generate, 
                user_prompt=prompt,  
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
                **kwargs,
            )
            # Assuming Unify's generate returns a string
            return [response]  

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

    # ... [Adapt other methods from the OpenAI code as needed] ...

    def unload_model(self):
        """
        Unloads the model and performs cleanup.
        """
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "_async_client" in self.__dict__:
            del self.__dict__["_async_client"]
        # TODO:  Add any Unify-specific cleanup logic 
        gc.collect()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()
        state.pop("retry_wrapper", None)
        state.pop("client", None)
        state.pop("_async_client", None)
        state["executor_pools"].clear()
        # TODO: Add any Unify-specific serialization logic if needed
        return state

__all__ = ["UnifyAI"]
