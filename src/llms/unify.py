import gc
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Union,
    cast,
    Dict,
)
import tiktoken 
from tiktoken import Encoding

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)

try:
    from unify.clients import Unify as UnifyClient, AsyncUnify as AsyncUnifyClient
except ImportError:
    raise ImportError(
        "Could not import unify client, please install it with `pip install unify-client`"
    )

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)

logger = logging.getLogger(__name__)


def _normalize_model_name(model_name: str):  # pragma: no cover
    """Normalize the model name to be consistent with tiktoken."""
    return model_name.replace("gpt-3.5-turbo", "gpt-3.5-turbo-0301")


def _is_chat_model(model_name: str):  # pragma: no cover
    """Check if the model is a chat model."""
    return "turbo" in model_name or "gpt-4" in model_name


def _is_gpt_3_5(model_name: str):  # pragma: no cover
    """Check if the model is a GPT-3.5 variant."""
    return "gpt-3.5" in model_name


def _is_gpt_3_5_legacy(model_name: str):  # pragma: no cover
    """Check if the model is a legacy GPT-3.5 variant."""
    return "gpt-3.5-turbo-0301" in model_name


def _is_128k_model(model_name: str):  # pragma: no cover
    return (
        model_name == "gpt-3.5-turbo-16k"
        or model_name == "gpt-3.5-turbo-16k-0613"
        or model_name == "gpt-4-32k"
        or model_name == "gpt-4-32k-0613"
    )


def _is_gpt_mini(model_name: str):  # pragma: no cover
    """Check if the model is a GPT-mini variant."""
    return "gpt-3.5-turbo-0613" in model_name or "gpt-4-0613" in model_name


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
        provider: Optional[str] = None,
        cache_folder_path: Optional[str] = None,
        # Initializes the UnifyAI instance.
        unify_client: Optional[UnifyClient] = None,
        async_unify_client: Optional[AsyncUnifyClient] = None,
        **kwargs,
    ):
        """
        Initializes the UnifyAI instance.

        unify_client: Optional[UnifyClient] = None
        async_unify_client: Optional[AsyncUnifyClient] = None

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
        self.base_url = base_url
        self.api_version = api_version
        self.additional_params = kwargs
        self.system_prompt = system_prompt
        self.provider = provider
        self.unify_client = unify_client
        self.async_unify_client = async_unify_client
        self.endpoint = kwargs.get("endpoint", None)
        self.model = kwargs.get("model", None)
        self.client_params = kwargs.get("client_params", {})
        self.executor_pools: Dict[int, ThreadPoolExecutor] = {}

    @property
    def client(self) -> UnifyClient:
        """
        Initializes and returns a synchronous Unify client.

        Returns:
            Unify: The synchronous Unify client.
        """
        if self.unify_client is None:
            _additional_params: Dict[str, Any] = {}
            if self.api_key:
                _additional_params["api_key"] = self.api_key
            if self.endpoint:
                _additional_params["endpoint"] = self.endpoint
            elif self.model and self.provider:
                _additional_params["model"] = self.model
                _additional_params["provider"] = self.provider
            if self.client_params:
                _additional_params.update(self.additional_params)
            self.unify_client = UnifyClient(**_additional_params)
        return self.unify_client

    @property
    def _async_client(self) -> AsyncUnifyClient:
        if self.async_unify_client is None:
            """
            Initializes and returns an asynchronous Unify client.

            Returns:
                AsyncUnify: The asynchronous Unify client.
            """
            _additional_params: Dict[str, Any] = {}
            if self.api_key:
                _additional_params["api_key"] = self.api_key
            if self.endpoint:
                _additional_params["endpoint"] = self.endpoint
            elif self.model and self.provider:
                _additional_params["model"] = self.model
                _additional_params["provider"] = self.provider
            if self.client_params:
                _additional_params.update(self.additional_params)
            self.async_unify_client = AsyncUnifyClient(**_additional_params)
        return self.async_unify_client

    @ring.lru(maxsize=5000)
    def tokenizer(self) -> Encoding:
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            logger.warning(
                f"Could not find encoding for model {self.model_name}. Using cl100k_base."
            )
            return tiktoken.get_encoding("cl100k_base")

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
        model_name = _normalize_model_name(self.model_name)
        format_tokens = 0
        if _is_chat_model(model_name):
            # Each message is up to 4 tokens and there are 3 messages
            # (system prompt, user prompt, assistant response)
            # and then we have to account for the system prompt
            format_tokens = 4 * 3 + self.count_tokens(
                cast(str, self.system_prompt)
            )
        if "32k" in model_name:
            max_context_length = 32768
        elif "16k" in model_name:
            max_context_length = 16384
        elif _is_128k_model(model_name):
            max_context_length = 128000
        elif _is_gpt_3_5(self.model_name):
            if _is_gpt_3_5_legacy(self.model_name):
                max_context_length = 4096
            else:
                max_context_length = 16385
        elif model_name.startswith("text-davinci"):
            max_context_length = 4097
        elif model_name.startswith("code-davinci"):
            max_context_length = 8001
        elif any(
            model_name.startswith(prefix)
            for prefix in ["text-curie", "text-babbage", "text-ada"]
        ) or model_name in ["ada", "babbage", "curie", "davinci"]:
            max_context_length = 2049
        else:
            max_context_length = 8192
        return max_context_length - max_new_tokens - format_tokens

    def _get_max_output_length(self) -> None | int:
        if _is_128k_model(self.model_name) and _is_gpt_mini(self.model_name):
            return 16384
        elif _is_128k_model(self.model_name) or (
            _is_gpt_3_5(self.model_name) and not (_is_gpt_3_5_legacy(self.model_name))
        ):
            return 4096
        else:
            return None

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        return len(self.tokenizer.encode(value))

    @retry(
        retry=retry_if_exception,
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        stop=stop_any(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
    )
    def retry_wrapper(self, func: Callable, **kwargs) -> Any:
        """Retry wrapper for API calls."""
        return func(**kwargs)

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
            Exception: If there's an error during API interaction.
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
                    func=self.client.generate,  # Access the protected method
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
                raise Exception(f"Error during Unify API call: {str(e)}")

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
        self.unify_client = None
        self.async_unify_client = None
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
