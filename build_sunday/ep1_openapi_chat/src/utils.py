"""
Utility helpers for the Q&A Bot.

This module implements:
1. Exponential backoff retry logic for asynchronous functions.
2. Explanations of core production concepts: Rate Limits, Retry logic, and Jitter.
"""

import asyncio
import random
import time
from typing import Callable, Any, TypeVar, Cast

# Import specific OpenAI exceptions so we can inspect them in our retry logic
import openai
from rich.console import Console

# Type variable to preserve the signatures of wrapped functions (Type Hinting best practice)
F = TypeVar('F', bound=Callable[..., Any])

console = Console()


def retry_on_api_error(
    max_retries: int = 3,
    min_delay: float = 2.0,
    max_delay: float = 10.0
) -> Callable[[F], F]:
    """
    Decorator that implements asynchronous exponential backoff with jitter.
    
    If the decorated function raises a retriable OpenAI API exception, it will
    wait and retry up to `max_retries` times before propagating the exception.
    
    WHY EXPONENTIAL BACKOFF & JITTER? (Educational Insight):
    ---------------------------------------------------------
    1. Exponential Backoff: When a server is overloaded or you hit a rate limit,
       retrying immediately just adds more load and causes consecutive failures.
       By multiplying the delay (e.g., 2s, 4s, 8s...), we give the remote server
       breathing room to recover or reset rate limit windows.
    2. Jitter (Randomized Delay): If thousands of bot clients retry at exactly
       the same time (e.g., exactly 2 seconds, then 4 seconds), they create a 
       wave of synchronized requests called the "Thundering Herd Problem".
       Adding a small random variance (jitter) spreads out the requests,
       increasing the likelihood that subsequent requests will succeed.
    """
    
    def decorator(func: F) -> F:
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = min_delay
            
            for attempt in range(1, max_retries + 2):  # +1 for original attempt, +1 for retry calculation bounds
                try:
                    return await func(*args, **kwargs)
                
                except (openai.APITimeoutError, openai.APIConnectionError) as e:
                    # Connection issues or timeouts are always retriable
                    if attempt > max_retries:
                        console.print(f"[bold red]❌ Connection error: Max retries ({max_retries}) reached. Failing...[/bold red]")
                        raise e
                    
                    # Calculate delay with jitter
                    # Delay grows exponentially: delay = min_delay * (2 ^ (attempt - 1))
                    # Jitter: randomized between 0.5 * delay and 1.5 * delay
                    current_delay = delay * (2 ** (attempt - 1))
                    jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                    jittered_delay = min(jittered_delay, max_delay)
                    
                    console.print(
                        f"[yellow]⚠️ Connection timeout/error (Attempt {attempt}/{max_retries}). "
                        f"Retrying in {jittered_delay:.2f}s... (Error: {e.__class__.__name__})[/yellow]"
                    )
                    await asyncio.sleep(jittered_delay)
                    
                except openai.RateLimitError as e:
                    # Rate limit (HTTP 429) is retriable
                    if attempt > max_retries:
                        console.print(f"[bold red]❌ Rate limit: Max retries ({max_retries}) reached. Failing...[/bold red]")
                        raise e
                    
                    current_delay = delay * (2 ** (attempt - 1))
                    jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                    jittered_delay = min(jittered_delay, max_delay)
                    
                    console.print(
                        f"[yellow]⚠️ Rate limit hit (Attempt {attempt}/{max_retries}). "
                        f"Retrying in {jittered_delay:.2f}s...[/yellow]"
                    )
                    await asyncio.sleep(jittered_delay)
                    
                except openai.APIStatusError as e:
                    # Specific HTTP status codes from OpenAI
                    # Do NOT retry:
                    # - 401: AuthenticationError (Invalid API Key)
                    # - 400: BadRequestError (Invalid prompt, invalid parameters)
                    # - 403: PermissionDeniedError (Blocked country or user)
                    # - 404: NotFoundError (Model doesn't exist)
                    if e.status_code in [400, 401, 403, 404]:
                        # Non-retriable developer/user errors
                        raise e
                    
                    # 5xx errors (Internal Server Errors) are retriable
                    if attempt > max_retries:
                        console.print(f"[bold red]❌ Server error: Max retries ({max_retries}) reached. Failing...[/bold red]")
                        raise e
                        
                    current_delay = delay * (2 ** (attempt - 1))
                    jittered_delay = random.uniform(current_delay * 0.5, current_delay * 1.5)
                    jittered_delay = min(jittered_delay, max_delay)
                    
                    console.print(
                        f"[yellow]⚠️ Server error {e.status_code} (Attempt {attempt}/{max_retries}). "
                        f"Retrying in {jittered_delay:.2f}s...[/yellow]"
                    )
                    await asyncio.sleep(jittered_delay)
                    
                except Exception as e:
                    # Catch-all for any other unanticipated exception (no retry)
                    raise e
                    
        return wrapper  # type: ignore

    return decorator
