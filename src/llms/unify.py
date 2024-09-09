import gc
import logging
import sys
import openai
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, List, Optional, Union, cast
from tiktoken import Encoding

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)

from unify.clients import AsyncUnify
from unify.clients import Unify as UnifyClient 

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)

DEFAULT_BATCH_SIZE = 10

@lru_cache(maxsize=None)
def _normalize_model_name(model_name: str) -> str:
    if ":" in model_name:  # pragma: no cover
        # Handles extracting the model name from a fine-tune
        # model name like "ft:babbage-002:org:datadreamer:xxxxxxx"
        model_name = model_name.split(":")[1]
    return model_name


@lru_cache(maxsize=None)
def _is_gpt_3(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(
        gpt3_name in model_name for gpt3_name in ["davinci", "ada", "curie", "gpt-3-"]
    )


@lru_cache(maxsize=None)
def _is_gpt_3_5(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt35_name in model_name for gpt35_name in ["gpt-3.5-", "gpt-35-"])


@lru_cache(maxsize=None)
def _is_gpt_3_5_legacy(model_name: str):
    model_name = _normalize_model_name(model_name)
    return _is_gpt_3_5(model_name) and (
        "-0613" in model_name
        or (_is_instruction_tuned(model_name) and not _is_chat_model(model_name))
    )


@lru_cache(maxsize=None)
def _is_gpt_4(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        model_name == "gpt-4"
        or any(gpt4_name in model_name for gpt4_name in ["gpt-4-"])
        or _is_gpt_4o(model_name)
    )


@lru_cache(maxsize=None)
def _is_gpt_4o(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt4_name in model_name for gpt4_name in ["gpt-4o"])


@lru_cache(maxsize=None)
def _is_gpt_mini(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt_mini_name in model_name for gpt_mini_name in ["-mini"])


@lru_cache(maxsize=None)
def _is_128k_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return _is_gpt_4(model_name) and (
        _is_gpt_4o(model_name) or "-preview" in model_name or "2024-04-09" in model_name
    )


@lru_cache(maxsize=None)
def _is_chat_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_gpt_3_5(model_name) or _is_gpt_4(model_name)
    ) and not model_name.endswith("-instruct")


@lru_cache(maxsize=None)
def _is_instruction_tuned(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_chat_model(model_name)
        or model_name.startswith("text-")
        or model_name.endswith("-instruct")
    )


class UnifyException(Exception):
    """Custom exception for UnifyAI class."""

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
        system_prompt: Optional[str] = None,
        organization: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        retry_on_fail: bool = True,
        cache_folder_path: Optional[str] = None,
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
            provider (str, optional): The name of the language model provider (e.g., "openai", "anthropic", etc.).
            **kwargs: Additional keyword arguments to pass to the Unify clients.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.organization = organization
        self.api_key = api_key
        if not self.api_key:
            raise APIKeyNotFoundError("Unify API key is required")
        self.base_url = base_url
        self.api_version = api_version
        if self.endpoint and (self.model or self.provider):
            raise ValueError("If endpoint is provided, model and provider should not be specified.")

        if self.endpoint:
            self.model, self.provider = self.endpoint.split("@")
        elif not (self.model and self.provider):
            raise ValueError("Both model and provider must be specified if endpoint is not provided.")
        self.additional_params = kwargs
        self.system_prompt = system_prompt
        if self.system_prompt is None and _is_chat_model(self.model_name):
            self.system_prompt = "You are a helpful assistant."
        self.provider = provider

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

        # Initialize the Unify clients
        self._client = self._get_client()
        self._async_client = self._get_async_client()

    @cached_property
    def retry_wrapper(self):
        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.InternalServerError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.APIError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(UnifyException),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.APIConnectionError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        def _retry_wrapper(func, **kwargs):
            return func(**kwargs)

        _retry_wrapper.__wrapped__.__module__ = None  # type: ignore[attr-defined]
        _retry_wrapper.__wrapped__.__qualname__ = f"{self.__class__.__name__}.run"  # type: ignore[attr-defined]
        return _retry_wrapper

    def _get_client(self) -> UnifyClient:
        """
        Initializes and returns a synchronous Unify client.

        Returns:
            UnifyClient: The synchronous Unify client.
        """
        try:
            return UnifyClient(
                api_key=self.api_key,
                endpoint=self.endpoint,
                model_name=self.model_name,
                provider=self.provider,
                **self.additional_params,
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
                endpoint=self.endpoint,
                model_name=self.model_name,
                provider=self.provider,
                **self.additional_params,
            )
        except Exception as e:
            raise UnifyException(f"Failed to initialize Async Unify client: {str(e)}")

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        This method should use the Unify API or a compatible tokenizer to accurately count tokens.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        # TODO (urgent): Replace placeholder with actual token counting using Unify API
        #  or a tokenizer that aligns with how Unify counts tokens.
        return len(value.split())

    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        """Gets the maximum context length for the model.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
        # TODO (urgent): Fetch the actual maximum context length from Unify.
        # This is a placeholder value - it might be inaccurate.
        return 4096 - max_new_tokens

    def _run_batch(
        self,
        max_length_func: Callable[[List[str]], int],
        inputs: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        repetition_penalty: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[List[str], List[List[str]]]:
        """
        Runs a batch of prompts through the Unify API.

        Args:
            max_length_func (Callable): Function to get the maximum length of a list of strings.
            inputs (List[str]): List of prompts to process.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
            temperature (float, optional): Sampling temperature (0.0 - 1.0).
            top_p (float, optional): Nucleus sampling probability (0.0 - 1.0).
            n (int, optional): Number of responses to generate per prompt.
            stop (str or List[str], optional): Stop sequence(s) for text generation.
            repetition_penalty (float, optional): Repetition penalty (1.0 - 2.0).
            logit_bias (Dict[int, float], optional): Logit bias for specific tokens.
            batch_size (int, optional): Batch size for API requests.
            seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments to pass to the Unify API.

        Returns:
            Union[List[str], List[List[str]]]: A list of generated responses.
                If `n` is 1, each element is a string.
                If `n` is greater than 1, each element is a list of strings (multiple responses per prompt).

        Raises:
            UnifyException: If there's an error during API interaction.
        """
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

        if batch_size not in self.executor_pools:
            self.executor_pools[batch_size] = ThreadPoolExecutor(
                max_workers=batch_size
            )

        # Wrap API call with retry logic
        def get_generated_texts(
            self: "UnifyAI", kwargs: dict, prompt: str
        ) -> List[str]:
            try:
                # Assuming Unify's API is similar to OpenAI's
                response = self.retry_wrapper(
                    func=self._client._generate,  # Access the protected method
                    user_message=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    n=n,
                    **kwargs,
                )

                # Assuming Unify returns generated text within a 'choices' list
                # Adapt based on the actual API response structure
                if isinstance(response, str):
                    return [response]
                elif hasattr(response, "choices"):
                    return [
                        choice.message.content.strip()
                        for choice in response.choices
                    ]
                else:
                    raise ValueError("Unexpected response format from Unify API")
            except Exception as e:
                raise UnifyException(
                    f"Error during Unify API call: {str(e)}"
                )

        # Use thread pool for parallel processing
        generated_texts_batch = list(
            self.executor_pools[batch_size].map(
                partial(get_generated_texts, self, kwargs), prompts
            )
        )

        return (
            [batch[0] for batch in generated_texts_batch]
            if n == 1
            else generated_texts_batch
        )

    def _run(
        self,
        prompts: Iterable[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        repetition_penalty: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: Optional[int] = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Union[Generator[str, None, None], List[str], List[List[str]]]:
        """
        Generates text using the Unify API, handling both streaming and non-streaming responses.

        Args:
            prompts (Iterable[str]): The prompts to generate text from.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
            temperature (float, optional): The sampling temperature (0.0 - 1.0).
            top_p (float, optional): The nucleus sampling probability (0.0 - 1.0).
            n (int, optional): The number of responses to generate for each prompt.
            stop (str or List[str], optional): Stop sequence(s) for text generation.
            repetition_penalty (float, optional): Repetition penalty for text generation.
            logit_bias (Dict[int, float], optional): Logit bias for specific tokens.
            batch_size (int, optional): The batch size for processing prompts.
            seed (int, optional): Random seed for reproducibility.
            return_generator (bool, optional): If True, returns a generator for streaming responses. 
                                              Otherwise, returns a list of responses.

        Returns:
            Union[Generator[str, None, None], List[str], List[List[str]]]: 
                - If `return_generator` is True, returns a generator yielding strings (streaming responses).
                - If `return_generator` is False and `n` is 1, returns a list of strings (one response per prompt).
                - If `return_generator` is False and `n` is greater than 1, returns a list of lists of strings (multiple responses per prompt).
        """
        if return_generator:

            def generator_wrapper():
                for prompt in prompts:
                    yield from self._run_batch(
                        max_length_func=lambda x: max(len(s) for s in x),
                        inputs=[prompt],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        stop=stop,
                        repetition_penalty=repetition_penalty,
                        logit_bias=logit_bias,
                        batch_size=1,  # Ensure batch size of 1 for streaming
                        seed=seed,
                        stream=True,  # Enable streaming for the batch
                        **kwargs,
                    )

            return generator_wrapper()
        else:
            return self._run_batch(
                max_length_func=lambda x: max(len(s) for s in x),
                inputs=list(prompts),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                repetition_penalty=repetition_penalty,
                logit_bias=logit_bias,
                batch_size=batch_size,
                seed=seed,
                **kwargs,
            )

    def unload_model(self):
        """Unloads the model and performs cleanup."""
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "_async_client" in self.__dict__:
            del self.__dict__["_async_client"]
        for pool in self.executor_pools.values():
            pool.shutdown()
        self.executor_pools.clear()
        gc.collect()

    def __getstate__(self) -> dict:
        """Gets the state of the object for serialization."""
        state = super().__getstate__()
        state.pop("retry_wrapper", None)
        state.pop("client", None)
        state.pop("_async_client", None)
        state["executor_pools"].clear()
        return state

__all__ = ["UnifyAI"]
