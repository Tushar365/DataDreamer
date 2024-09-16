import os
import unittest
from unittest.mock import MagicMock, patch

from langchain.llms import UnifyAI


class TestUnifyAI(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["UNIFY_API_KEY"] = "test_api_key"
        self.model_name = "test_model"
        self.system_prompt = "test_system_prompt"
        self.n = 2
        self.temperature = 0.7
        self.top_p = 0.8
        self.stop = ["test_stop"]
        self.max_new_tokens = 10
        self.inputs = ["test_input1", "test_input2"]
        self.mock_response = MagicMock()
        self.mock_response.choices = [
            MagicMock(message={"content": "test_response1"}),
            MagicMock(message={"content": "test_response2"}),
        ]

    def tearDown(self) -> None:
        os.environ.pop("UNIFY_API_KEY", None)

    @patch("langchain.llms.unify.UnifyClient")
    def test_init(self, mock_unify_client):
        llm = UnifyAI(model_name=self.model_name)
        self.assertEqual(llm.model_name, self.model_name)
        self.assertIsNone(llm.system_prompt)
        self.assertEqual(llm.client, mock_unify_client.return_value)

    @patch("langchain.llms.unify.UnifyClient")
    def test_init_with_system_prompt(self, mock_unify_client):
        llm = UnifyAI(
            model_name=self.model_name, system_prompt=self.system_prompt
        )
        self.assertEqual(llm.system_prompt, self.system_prompt)

    @patch("langchain.llms.unify.UnifyClient")
    def test_count_tokens(self, mock_unify_client):
        llm = UnifyAI(model_name=self.model_name)
        # TODO: Update this test when token counting is implemented for UnifyAI
        self.assertEqual(
            llm.count_tokens("test_string"), len("test_string".split())
        )

    @patch("langchain.llms.unify.UnifyClient")
    def test_get_max_context_length(self, mock_unify_client):
        llm = UnifyAI(model_name=self.model_name)
        # TODO: Update this test when get_max_context_length is implemented
        #  for UnifyAI
        self.assertEqual(
            llm.get_max_context_length(max_new_tokens=10), 4096 - 10
        )

    @patch("langchain.llms.unify.UnifyAI.retry_wrapper")
    def test__run_batch(self, mock_retry_wrapper):
        llm = UnifyAI(model_name=self.model_name)
        mock_retry_wrapper.return_value = self.mock_response

        # Test with n=1
        results = llm._run_batch(
            max_length_func=lambda x: max(len(s) for s in x),
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stop=self.stop,
        )
        self.assertEqual(len(results), len(self.inputs))
        self.assertIsInstance(results[0], str)

        # Test with n>1
        results = llm._run_batch(
            max_length_func=lambda x: max(len(s) for s in x),
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.stop,
        )
        self.assertEqual(len(results), len(self.inputs))
        self.assertIsInstance(results[0], list)
        self.assertEqual(len(results[0]), self.n)

    @patch("langchain.llms.unify.UnifyAI.retry_wrapper")
    def test__run_batch_with_string_response(self, mock_retry_wrapper):
        """Test _run_batch when Unify API returns a string (not in a list)."""
        llm = UnifyAI(model_name=self.model_name)
        mock_retry_wrapper.return_value = "test_response"
        results = llm._run_batch(
            max_length_func=lambda x: max(len(s) for s in x),
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stop=self.stop,
        )
        self.assertEqual(len(results), len(self.inputs))
        self.assertIsInstance(results[0], str)

    @patch("langchain.llms.unify.UnifyAI._run_batch")
    def test__run(self, mock__run_batch):
        llm = UnifyAI(model_name=self.model_name)
        mock__run_batch.return_value = ["test_response"]

        # Test with return_generator=False
        results = llm._run(
            prompts=self.inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stop=self.stop,
        )
        self.assertIsInstance(results, list)

        # Test with return_generator=True
        results = llm._run(
            prompts=self.inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stop=self.stop,
            return_generator=True,
        )
        self.assertIsInstance(results, Generator)

    @patch("langchain.llms.unify.UnifyClient")
    def test_unload_model(self, mock_unify_client):
        llm = UnifyAI(model_name=self.model_name)
        llm.unload_model()
        self.assertIsNone(llm.unify_client)

    def test__getstate__(self):
        llm = UnifyAI(model_name=self.model_name)
        state = llm.__getstate__()
        self.assertNotIn("retry_wrapper", state)
        self.assertNotIn("client", state)
        self.assertNotIn("_async_client", state)


if __name__ == "__main__":
    unittest.main()
