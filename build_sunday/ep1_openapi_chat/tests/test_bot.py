"""
Unit tests for Build Sunday Q&A Bot.

To run these tests, navigate to the EP_1_SMARTQ&A directory and run:
`PYTHONPATH=. pytest tests/`
"""

import os
import pytest
from unittest import mock

# Import classes to test
from src.config import AppConfig, ConfigValidationError
from src.token_counter import TokenTracker
from src.conversation import ConversationHistory


# ==============================================================================
# Configuration Tests
# ==============================================================================

def test_config_load_success():
    """Test that configurations load correctly when valid environment variables are present."""
    mock_env = {
        "OPENAI_API_KEY": "sk-mock-key-12345",
        "DEFAULT_MODEL": "gpt-4o",
        "MAX_COMPLETION_TOKENS": "512",
        "TEMPERATURE": "0.5",
        "MAX_RETRIES": "5",
        "RETRY_MIN_DELAY": "1.0",
        "RETRY_MAX_DELAY": "5.0",
        "API_TIMEOUT": "15.0"
    }
    with mock.patch.dict(os.environ, mock_env, clear=True):
        config = AppConfig.load_from_env()
        assert config.openai_api_key == "sk-mock-key-12345"
        assert config.default_model == "gpt-4o"
        assert config.max_completion_tokens == 512
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.retry_min_delay == 1.0
        assert config.retry_max_delay == 5.0
        assert config.api_timeout == 15.0


def test_config_load_missing_key():
    """Test that omitting the API key raises a ConfigValidationError."""
    mock_env = {
        "DEFAULT_MODEL": "gpt-4o-mini"
    }
    with mock.patch.dict(os.environ, mock_env, clear=True):
        with pytest.raises(ConfigValidationError) as excinfo:
            AppConfig.load_from_env()
        assert "OPENAI_API_KEY is not set" in str(excinfo.value)


def test_config_load_placeholder_key():
    """Test that keeping the placeholder API key raises a ConfigValidationError."""
    mock_env = {
        "OPENAI_API_KEY": "your-openai-api-key-here"
    }
    with mock.patch.dict(os.environ, mock_env, clear=True):
        with pytest.raises(ConfigValidationError) as excinfo:
            AppConfig.load_from_env()
        assert "placeholder value" in str(excinfo.value)


def test_config_invalid_data_types_fallback():
    """Test that invalid types for integers/floats fall back to defaults gracefully."""
    mock_env = {
        "OPENAI_API_KEY": "sk-valid-key",
        "MAX_COMPLETION_TOKENS": "not-an-int",
        "TEMPERATURE": "not-a-float",
        "MAX_RETRIES": "not-an-int",
        "RETRY_MIN_DELAY": "not-a-float",
        "RETRY_MAX_DELAY": "not-a-float",
        "API_TIMEOUT": "not-a-float"
    }
    with mock.patch.dict(os.environ, mock_env, clear=True):
        config = AppConfig.load_from_env()
        # Verify defaults are used
        assert config.max_completion_tokens == 1024
        assert config.temperature == 0.7
        assert config.max_retries == 3
        assert config.retry_min_delay == 2.0
        assert config.retry_max_delay == 10.0
        assert config.api_timeout == 30.0


def test_config_load_local_endpoint():
    """Test that local endpoints bypass API key requirement and set mock defaults."""
    mock_env = {
        "API_BASE_URL": "http://localhost:11434/v1",
        "DEFAULT_MODEL": "llama2",
    }
    with mock.patch.dict(os.environ, mock_env, clear=True):
        config = AppConfig.load_from_env()
        assert config.api_base_url == "http://localhost:11434/v1"
        assert config.openai_api_key == "local-ollama"
        assert config.default_model == "llama2"


# ==============================================================================
# Token Tracker Tests
# ==============================================================================

def test_token_tracker_string_counting():
    """Verify that string token counts match basic tiktoken expectations."""
    tracker = TokenTracker(model_name="gpt-4o-mini")
    empty_tokens = tracker.count_string_tokens("")
    hello_tokens = tracker.count_string_tokens("Hello world")
    
    assert empty_tokens == 0
    assert hello_tokens > 0


def test_token_tracker_message_counting():
    """Verify message structures are calculated with standard token overheads."""
    tracker = TokenTracker(model_name="gpt-4o-mini")
    messages = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "Hello"}
    ]
    
    token_count = tracker.count_messages_tokens(messages)
    # The count should include:
    # - "You are a bot." tokens
    # - "Hello" tokens
    # - role names tokens ("system", "user")
    # - formatting/overhead tokens (3 tokens per message + 3 tokens reply priming)
    assert token_count > len(messages) * 3 + 3


def test_token_tracker_cost_calculation():
    """Verify that cost calculation multiplies token rates correctly."""
    tracker = TokenTracker(model_name="gpt-4o-mini")
    
    # 1,000,000 prompt tokens = $0.15
    # 1,000,000 completion tokens = $0.60
    cost = tracker.calculate_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 0.75) < 1e-6


# ==============================================================================
# Conversation History Tests
# ==============================================================================

def test_conversation_history_lifecycle():
    """Test standard adding and clearing of messages in ConversationHistory."""
    sys_prompt = "You are testing."
    history = ConversationHistory(system_prompt=sys_prompt)
    
    # Initial state should only contain the system prompt
    msgs = history.get_messages()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": sys_prompt}
    
    # Add exchange
    history.add_user_message("Hello")
    history.add_assistant_message("Hi there")
    
    msgs = history.get_messages()
    assert len(msgs) == 3
    assert msgs[1] == {"role": "user", "content": "Hello"}
    assert msgs[2] == {"role": "assistant", "content": "Hi there"}
    
    # Clear session
    history.clear()
    msgs = history.get_messages()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": sys_prompt}


def test_conversation_history_truncation():
    """Test that older messages are discarded when token capacity is exceeded, keeping system prompt."""
    sys_prompt = "Short system prompt."
    # We set max_history_tokens low to force truncation
    history = ConversationHistory(
        system_prompt=sys_prompt,
        max_history_tokens=50
    )
    
    # Add a series of exchanges
    history.add_user_message("This is a very long user query designed to fill up the token budget quickly.")
    history.add_assistant_message("This is the response from the assistant, which also consumes token limits.")
    
    history.add_user_message("Here is another user query that we will add.")
    history.add_assistant_message("And here is the second assistant response.")
    
    msgs = history.get_messages()
    
    # Ensure system prompt is preserved
    assert msgs[0] == {"role": "system", "content": sys_prompt}
    
    # Ensure some truncation took place (we should not have all 4 messages + system prompt)
    assert len(msgs) < 5
