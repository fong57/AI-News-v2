"""Tests for LangSmith tracing configuration."""

import os
import pytest

from litenews.config.settings import Settings
from litenews.config.tracing import (
    setup_tracing,
    disable_tracing,
    get_tracing_status,
    test_langsmith_connection,
)


class TestTracingConfiguration:
    """Tests for tracing setup and configuration."""
    
    def test_tracing_disabled_by_default(self):
        """Test that tracing is disabled when no API key is set."""
        settings = Settings(langchain_tracing_v2=True, langchain_api_key="")
        assert not settings.is_tracing_enabled()
    
    def test_tracing_enabled_with_key(self):
        """Test that tracing is enabled with API key."""
        settings = Settings(
            langchain_tracing_v2=True,
            langchain_api_key="test_key_12345",
        )
        assert settings.is_tracing_enabled()
    
    def test_tracing_disabled_when_flag_false(self):
        """Test that tracing is disabled when flag is false."""
        settings = Settings(
            langchain_tracing_v2=False,
            langchain_api_key="test_key_12345",
        )
        assert not settings.is_tracing_enabled()
    
    def test_setup_tracing_sets_env_vars(self):
        """Test that setup_tracing sets environment variables."""
        settings = Settings(
            langchain_tracing_v2=True,
            langchain_api_key="test_key_12345",
            langchain_project="test-project",
        )
        
        result = setup_tracing(settings)
        
        assert result is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "test_key_12345"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"
        
        disable_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
    
    def test_get_tracing_status(self):
        """Test getting tracing status."""
        settings = Settings(
            langchain_tracing_v2=True,
            langchain_api_key="test_key",
            langchain_project="my-project",
        )
        
        from litenews.config.settings import get_settings
        get_settings.cache_clear()
        
        status = get_tracing_status()
        assert "enabled" in status
        assert "project" in status


class TestLangSmithConnection:
    """Tests for LangSmith API connection."""
    
    @pytest.fixture
    def api_key(self):
        """Get LangSmith API key from environment."""
        key = os.environ.get("LANGCHAIN_API_KEY", "")
        if not key or key == "your_langsmith_api_key_here":
            pytest.skip("LANGCHAIN_API_KEY not set")
        return key
    
    def test_connection(self, api_key):
        """Test LangSmith API connection."""
        result = test_langsmith_connection(api_key)
        assert result["status"] == "success", f"Connection failed: {result.get('error')}"
    
    def test_invalid_key(self):
        """Test that invalid key returns error."""
        result = test_langsmith_connection("invalid_key_12345")
        assert result["status"] == "error"
