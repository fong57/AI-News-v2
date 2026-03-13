"""Tests for LLM integrations."""

import os
import pytest

from litenews.config.settings import Settings
from litenews.llms.perplexity import test_perplexity_connection
from litenews.llms.qwen import test_qwen_connection
from litenews.llms.base import get_llm


class TestPerplexityIntegration:
    """Tests for Perplexity LLM integration."""
    
    @pytest.fixture
    def api_key(self):
        """Get Perplexity API key from environment."""
        key = os.environ.get("PPLX_API_KEY", "")
        if not key or key == "your_perplexity_api_key_here":
            pytest.skip("PPLX_API_KEY not set")
        return key
    
    def test_connection(self, api_key):
        """Test basic Perplexity API connection."""
        result = test_perplexity_connection(api_key, model="sonar")
        assert result["status"] == "success", f"Connection failed: {result.get('error')}"
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_async_invoke(self, api_key):
        """Test async invocation."""
        settings = Settings(pplx_api_key=api_key, perplexity_model="sonar")
        llm = get_llm("perplexity", settings)
        
        response = await llm.ainvoke("What is 1+1? Reply with just the number.")
        assert response.content is not None
        assert "2" in response.content


class TestQwenIntegration:
    """Tests for Qwen/DashScope LLM integration."""
    
    @pytest.fixture
    def api_key(self):
        """Get DashScope API key from environment."""
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key or key == "your_dashscope_api_key_here":
            pytest.skip("DASHSCOPE_API_KEY not set")
        return key
    
    def test_connection(self, api_key):
        """Test basic Qwen API connection."""
        result = test_qwen_connection(api_key, model="qwen-turbo")
        assert result["status"] == "success", f"Connection failed: {result.get('error')}"
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_async_invoke(self, api_key):
        """Test async invocation."""
        settings = Settings(dashscope_api_key=api_key, qwen_model="qwen-turbo")
        llm = get_llm("qwen", settings)
        
        response = await llm.ainvoke("What is 1+1? Reply with just the number.")
        assert response.content is not None
        assert "2" in response.content


class TestLLMFactory:
    """Tests for the LLM factory function."""
    
    def test_get_llm_perplexity(self):
        """Test getting Perplexity LLM."""
        settings = Settings(pplx_api_key="test_key")
        llm = get_llm("perplexity", settings)
        assert llm.config.provider == "perplexity"
    
    def test_get_llm_qwen(self):
        """Test getting Qwen LLM."""
        settings = Settings(dashscope_api_key="test_key")
        llm = get_llm("qwen", settings)
        assert llm.config.provider == "qwen"
    
    def test_get_llm_invalid(self):
        """Test invalid provider raises error."""
        settings = Settings()
        with pytest.raises(ValueError):
            get_llm("invalid_provider", settings)
