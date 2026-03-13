"""LLM integrations for LiteNews AI."""

from litenews.llms.base import BaseLLM, get_llm
from litenews.llms.perplexity import PerplexityLLM
from litenews.llms.qwen import QwenLLM

__all__ = ["BaseLLM", "get_llm", "PerplexityLLM", "QwenLLM"]
