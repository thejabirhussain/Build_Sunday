"""
Configuration Module for Build Sunday Q&A Bot.

This module is responsible for:
1. Loading environment variables securely using `python-dotenv`.
2. Validating required configurations (like the OpenAI API key) on startup.
3. Defining default system settings and fallback values.
4. Explaining WHY environment variables are essential in production engineering.

WHY USE ENVIRONMENT VARIABLES? (Educational Insight):
---------------------------------------------------
1. Security: Storing secrets (like API keys) directly in code is a primary source of data leaks.
   If code is pushed to GitHub, your keys become public instantly, allowing others to run up bills.
2. Configuration Separation: The "Twelve-Factor App" methodology dictates that configuration
   should be separated from code. This lets you run the same code in Local, Staging, and Production
   environments simply by changing the environment variables, without modifying a single line of code.
3. Git Hygiene: By keeping variables in a .env file and adding .env to .gitignore, we keep our
   secrets off remote git repositories.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Define the root of the project to locate the .env file correctly
# regardless of where the runner script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the .env file
# By specifying the path, we ensure that running the app from different subfolders
# doesn't prevent python-dotenv from finding the configuration file.
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path)


class ConfigValidationError(ValueError):
    """Custom exception raised when application settings fail validation."""
    pass


@dataclass(frozen=True)
class AppConfig:
    """
    Centralized configurations for the Smart Q&A Application.
    
    Using a dataclass keeps settings immutable (frozen=True) and structured,
    preventing runtime accidents where a module inadvertently overwrites a setting.
    """
    openai_api_key: str
    api_base_url: Optional[str]
    default_model: str
    max_completion_tokens: int
    temperature: float
    max_retries: int
    retry_min_delay: float
    retry_max_delay: float
    api_timeout: float

    @classmethod
    def load_from_env(cls) -> "AppConfig":
        """
        Loads and validates environment variables, returning a validated AppConfig instance.
        
        Raises:
            ConfigValidationError: If required settings are missing or formatted incorrectly.
        """
        api_base_url = os.getenv("API_BASE_URL", "").strip()
        if not api_base_url:
            api_base_url = None

        # Check if the target endpoint is running locally
        is_local = api_base_url is not None and ("localhost" in api_base_url or "127.0.0.1" in api_base_url)

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        
        if not is_local:
            # Validation 1: API Key must be present for remote OpenAI requests
            if not api_key:
                raise ConfigValidationError(
                    "OPENAI_API_KEY is not set in the environment or .env file.\n"
                    "Please copy .env.example to .env and add your OpenAI API key."
                )
                
            # Validation 2: Ensure the user hasn't left the default placeholder
            if api_key == "your-openai-api-key-here":
                raise ConfigValidationError(
                    "OPENAI_API_KEY is still set to the placeholder value.\n"
                    "Please replace it with a valid OpenAI API key in your .env file."
                )
        else:
            # If running against a local OpenAI-compatible endpoint (like Ollama or vLLM),
            # default to a dummy key to prevent the OpenAI SDK from throwing validation errors.
            if not api_key or api_key == "your-openai-api-key-here":
                api_key = "local-ollama"

        # 3. Read and convert options with robust fallbacks
        default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini").strip()
        
        try:
            max_completion_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))
        except ValueError:
            max_completion_tokens = 1024
            
        try:
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
        except ValueError:
            temperature = 0.7

        try:
            max_retries = int(os.getenv("MAX_RETRIES", "3"))
        except ValueError:
            max_retries = 3

        try:
            retry_min_delay = float(os.getenv("RETRY_MIN_DELAY", "2.0"))
        except ValueError:
            retry_min_delay = 2.0

        try:
            retry_max_delay = float(os.getenv("RETRY_MAX_DELAY", "10.0"))
        except ValueError:
            retry_max_delay = 10.0

        try:
            api_timeout = float(os.getenv("API_TIMEOUT", "30.0"))
        except ValueError:
            api_timeout = 30.0

        return cls(
            openai_api_key=api_key,
            api_base_url=api_base_url,
            default_model=default_model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_min_delay=retry_min_delay,
            retry_max_delay=retry_max_delay,
            api_timeout=api_timeout
        )


# Global instance of configuration to be imported across modules
# Lazy validation occurs when this module is run or when components initialize.
# We will wrap the loading inside a helper function to facilitate clean startup logs.
def get_config() -> AppConfig:
    """Return the validated configuration. Helper function to capture errors cleanly."""
    return AppConfig.load_from_env()
