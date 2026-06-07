"""
API Client Module for Build Sunday Q&A Bot.

This module is responsible for:
1. Defining a clean abstraction (`BaseLLMClient`) for switching LLM providers in the future.
2. Initializing the `AsyncOpenAI` client.
3. Implementing async completion streams.
4. Handling OpenAI-specific exceptions and error logging.
5. Providing resilient retry loops specifically tailored for streaming generator outputs.

THE STREAMING & RETRY GOTCHA (Educational Insight):
--------------------------------------------------
1. Async Generators vs. Standard Async Functions: Standard retry decorators wrapping async 
   functions (using `await func()`) execute immediately. However, async generator functions 
   (using `yield`) do not execute code until the caller starts iterating over the generator.
2. The Problem: A decorator that wraps the function call will succeed instantly by returning the 
   generator object, but it will fail to catch errors that occur *during* iteration (e.g. midway 
   through a long response stream).
3. The Solution: In this client, we implement an explicit, clean generator-safe retry loop 
   directly inside our streaming methods. This ensures that if a network glitch occurs before the 
   stream starts, we retry the request creation cleanly.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any

import openai
from openai import AsyncOpenAI
from rich.console import Console

from src.config import AppConfig

console = Console()


class BaseLLMClient(ABC):
    """
    Abstract Base Class defining the LLM Gateway interface.
    
    Adhering to the Dependency Inversion Principle, this interface allows us to swap 
    OpenAI with Anthropic, Google Gemini, or local models (like Ollama) in `main.py`
    without changing the orchestration logic.
    """

    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Yields text chunks as they arrive from the LLM provider.
        
        Args:
            messages: The list of chat messages (history).
            model: Optional override for the target model.
            temperature: Optional override for creativity setting.
            max_tokens: Optional override for output token limits.
            
        Yields:
            str: Each chunk of text response.
        """
        pass


class OpenAIClient(BaseLLMClient):
    """
    Concrete implementation of BaseLLMClient interfacing with the OpenAI API.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the AsyncOpenAI client with credentials.
        
        Args:
            config: An instance of AppConfig containing key settings.
        """
        self.config = config
        
        # Initialize client with specified base URL, timeout and API Key
        self.client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.api_base_url,
            timeout=self.config.api_timeout
        )

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Requests completion from OpenAI and yields text tokens one-by-one.
        
        Handles API errors, rate limiting, and timeouts gracefully with a custom 
        generator-safe exponential backoff retry loop.
        """
        # Set runtime defaults from the configuration if overrides aren't provided
        target_model = model or self.config.default_model
        target_temp = temperature if temperature is not None else self.config.temperature
        target_max_tokens = max_tokens or self.config.max_completion_tokens

        delay = self.config.retry_min_delay
        max_retries = self.config.max_retries

        # Attempt tracking loop (attempt 0 is the initial try, up to max_retries extra attempts)
        for attempt in range(1, max_retries + 2):
            try:
                # Initiate the asynchronous completion stream
                response_stream = await self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,  # type: ignore
                    temperature=target_temp,
                    max_tokens=target_max_tokens,
                    stream=True
                )
                
                # Consume and yield each chunk's content text
                async for chunk in response_stream:
                    # Delta can be empty or have null content (e.g. first/last chunks, role assignments)
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                
                # If we iterated through the stream successfully, we break the retry loop
                return

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                if attempt > max_retries:
                    console.print(f"\n[bold red]❌ Connection error: Max retries ({max_retries}) reached. Failing...[/bold red]")
                    raise e
                
                # Jittered exponential backoff delay calculation
                current_delay = delay * (2 ** (attempt - 1))
                jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                jittered_delay = min(jittered_delay, self.config.retry_max_delay)

                console.print(
                    f"\n[yellow]⚠️ Connection timeout/error (Attempt {attempt}/{max_retries}). "
                    f"Retrying in {jittered_delay:.2f}s...[/yellow]"
                )
                await asyncio.sleep(jittered_delay)

            except openai.RateLimitError as e:
                if attempt > max_retries:
                    console.print(f"\n[bold red]❌ Rate limit: Max retries ({max_retries}) reached. Failing...[/bold red]")
                    raise e

                current_delay = delay * (2 ** (attempt - 1))
                jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                jittered_delay = min(jittered_delay, self.config.retry_max_delay)

                console.print(
                    f"\n[yellow]⚠️ OpenAI Rate Limit hit (Attempt {attempt}/{max_retries}). "
                    f"Retrying in {jittered_delay:.2f}s...[/yellow]"
                )
                await asyncio.sleep(jittered_delay)

            except openai.AuthenticationError as e:
                console.print("\n[bold red]❌ OpenAI Authentication Error: Invalid API Key. Please check your .env file.[/bold red]")
                raise e

            except openai.BadRequestError as e:
                console.print(f"\n[bold red]❌ Invalid Request Error: {e.message}[/bold red]")
                raise e

            except openai.APIStatusError as e:
                # Check for other HTTP server errors (5xx)
                if e.status_code >= 500:
                    if attempt > max_retries:
                        console.print(f"\n[bold red]❌ OpenAI Server Error: Max retries reached. Failing...[/bold red]")
                        raise e
                    
                    current_delay = delay * (2 ** (attempt - 1))
                    jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                    jittered_delay = min(jittered_delay, self.config.retry_max_delay)

                    console.print(
                        f"\n[yellow]⚠️ OpenAI Server Error {e.status_code} (Attempt {attempt}/{max_retries}). "
                        f"Retrying in {jittered_delay:.2f}s...[/yellow]"
                    )
                    await asyncio.sleep(jittered_delay)
                else:
                    # Let other status codes raise directly without retries (e.g. 403, 404)
                    raise e
                    
            except Exception as e:
                console.print(f"\n[bold red]❌ Unexpected exception during call: {str(e)}[/bold red]")
                raise e
