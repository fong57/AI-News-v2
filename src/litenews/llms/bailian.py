"""Alibaba Bailian (Model Studio) LLM via OpenAI-compatible API."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from litenews.config.llm_config import LLMConfig
from litenews.llms.base import BaseLLM


class BailianLLM(BaseLLM):
    """Bailian models through DashScope compatible-mode (OpenAI API shape).

    Use the model names shown in the Bailian console (e.g. ``qwen-plus``, ``qwen-max``).
    For international deployments, set ``BAILIAN_API_BASE`` to the regional compatible endpoint.
    """

    def _create_model(self) -> BaseChatModel:
        base = (self.config.api_base or "").strip()
        if not base:
            raise ValueError("Bailian requires a non-empty api_base (set BAILIAN_API_BASE).")
        return ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=base,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )


def test_bailian_connection(
    api_key: str,
    *,
    model: str = "qwen-plus",
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> dict:
    """Test Bailian compatible-mode API connectivity."""
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base.strip(),
            max_tokens=100,
        )
        response = llm.invoke("What is 2+2? Reply in one word.")
        return {
            "status": "success",
            "model": model,
            "response": response.content,
        }
    except Exception as e:
        return {
            "status": "error",
            "model": model,
            "error": str(e),
        }
