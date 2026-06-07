"""
Main Entry Point & CLI Loop for Build Sunday Q&A Bot.

This module is responsible for:
1. Initializing and orchestrating all components (`config`, `api_client`, `conversation`, `token_counter`).
2. Constructing an interactive CLI loop using the `rich` library.
3. Handling token streaming visually in the terminal.
4. Gracefully managing user interruptions (like Ctrl+C or Ctrl+D) and system errors.
5. Explaining production patterns in event-loop orchestration.

DEVELOPER EXPERIENCE (DX) INSIGHT:
---------------------------------
1. Interrupthandling: If a user presses Ctrl+C while the bot is generating a response,
   the app should NOT crash. Instead, it should immediately stop streaming, save the partial
   response, and return control to the user for the next prompt.
2. If they press Ctrl+C while waiting for a prompt, we interpret it as a command to exit.
3. Structured Output: We use `rich` panels, rules, and tables to make the CLI feel premium
   and professional, avoiding the plain, uninspiring terminal text format.
"""

import sys
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

# Import our custom modules
from src.config import get_config, ConfigValidationError
from src.token_counter import TokenTracker
from src.conversation import ConversationHistory
from src.api_client import OpenAIClient

# Initialize Rich Console for styling output
console = Console()


async def run_chat_loop() -> None:
    """
    Main asynchronous interactive chat loop.
    
    Orchestrates configuration checking, instantiation of services,
    prompt reading, API streaming, and statistics rendering.
    """
    # 1. Load and validate Configuration
    try:
        config = get_config()
    except ConfigValidationError as e:
        console.print(Panel.fit(
            f"[bold red]Configuration Validation Error:[/bold red]\n{str(e)}",
            title="⚠️ Startup Failure",
            border_style="red"
        ))
        sys.exit(1)

    # 2. Initialize core modules
    # We use a single shared TokenTracker instance for consistent model costing
    token_tracker = TokenTracker(model_name=config.default_model)
    
    # We set a context safety limit of 4096 tokens for history tracking
    conversation = ConversationHistory(
        system_prompt="You are a helpful, concise, and smart Q&A assistant for the YouTube series 'Build Sunday'. Be engaging, clear, and educational.",
        max_history_tokens=4096,
        token_tracker=token_tracker
    )
    
    client = OpenAIClient(config=config)

    # 3. Print Welcome Banner
    welcome_text = Text()
    welcome_text.append("🚀 Welcome to Build Sunday - Smart Q&A Bot!\n", style="bold cyan")
    welcome_text.append("A production-minded, clean architecture AI assistant.\n\n", style="italic white")
    welcome_text.append(f"• Model: [bold green]{config.default_model}[/bold green]\n", style="white")
    welcome_text.append(f"• Temperature: [bold green]{config.temperature}[/bold green]\n", style="white")
    welcome_text.append("• Commands: Type [bold yellow]exit[/bold yellow] or [bold yellow]clear[/bold yellow] to manage the session.\n", style="white")
    welcome_text.append("• Interruption: Press [bold yellow]Ctrl+C[/bold yellow] during output to stop generation.", style="white")
    
    console.print(Panel(
        welcome_text,
        title="[bold magenta]BUILD SUNDAY[/bold magenta]",
        border_style="magenta",
        expand=False
    ))

    # 4. Interactive Loop
    while True:
        try:
            # Display user prompt symbol
            console.print("\n[bold cyan]👤 You:[/bold cyan]")
            
            # Read input using standard blocking input inside a run_in_executor
            # block to keep the event loop from being blocked, though standard input()
            # is fine for simple CLI apps. We wrap it in a try block to handle Ctrl+D.
            user_input = await asyncio.to_thread(input, "> ")
            user_input = user_input.strip()

            # Handle Empty inputs
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[bold magenta]👋 Goodbye! Thanks for watching Build Sunday![/bold magenta]")
                break
                
            if user_input.lower() == "clear":
                conversation.clear()
                console.print("[green]✨ Conversation history cleared![/green]")
                continue

            # 5. Process prompt & calculate prompt tokens
            conversation.add_user_message(user_input)
            
            # Get current formatted history for the API call (which now includes the user's prompt)
            messages = conversation.get_messages()
            
            # Count the tokens of the outgoing payload
            prompt_tokens = token_tracker.count_messages_tokens(messages)

            # 6. Stream Assistant completion live
            console.print("\n[bold magenta]🤖 Assistant:[/bold magenta]")
            
            assistant_response = ""
            
            # Live context manager from Rich updates the terminal display dynamically
            # as tokens stream in.
            with Live("", console=console, auto_refresh=False) as live:
                try:
                    stream = client.stream_chat_completion(
                        messages=messages,
                        model=config.default_model,
                        temperature=config.temperature
                    )
                    
                    async for token in stream:
                        assistant_response += token
                        # Update the live terminal output
                        live.update(assistant_response)
                        live.refresh()
                        
                except asyncio.CancelledError:
                    # Catch event loop cancels
                    console.print("\n[yellow]⚠️ Generation cancelled by host event loop.[/yellow]")
                    raise
                except KeyboardInterrupt:
                    # User hit Ctrl+C during streaming! Stop and save the partial response.
                    console.print("\n[yellow]⏹️ Generation interrupted by user (Ctrl+C).[/yellow]")
                except Exception as e:
                    # Errors are already logged inside api_client.py, but we capture propagation
                    console.print(f"\n[bold red]❌ Failed to retrieve full completion: {str(e)}[/bold red]")
                    # Clean up last user message since response failed to generate
                    # to keep the stateless message logs in sync.
                    if len(conversation._messages) > 1:
                        conversation._messages.pop()
                    continue

            # If we received a response (partial or full), record it and save to memory
            if assistant_response:
                conversation.add_assistant_message(assistant_response)
                
                # Calculate completion tokens
                completion_tokens = token_tracker.count_string_tokens(assistant_response)
                
                # Record usage in tracker (updates cost and cumulative stats)
                step_cost = token_tracker.record_usage(prompt_tokens, completion_tokens)
                
                # Render usage statistics for this turn
                render_stats_table(prompt_tokens, completion_tokens, step_cost, token_tracker)

        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            # User hit Ctrl+C or Ctrl+D during the prompt input phase
            console.print("\n\n[bold magenta]👋 Goodbye! Thanks for watching Build Sunday![/bold magenta]")
            break
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Critical App Error: {str(e)}[/bold red]")
            break


def render_stats_table(
    prompt_tokens: int,
    completion_tokens: int,
    step_cost: float,
    tracker: TokenTracker
) -> None:
    """Prints a beautiful summary table showing token metrics & cost estimates."""
    stats = tracker.get_session_stats()
    
    table = Table(
        title="📊 Usage Metrics & Cost Estimation",
        title_style="bold dim white",
        caption="Estimated using local tiktoken analysis",
        caption_style="italic dim",
        box=None,
        padding=(0, 2)
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Current Turn", style="magenta", justify="right")
    table.add_column("Session Total", style="green", justify="right")
    
    table.add_row(
        "Prompt (Input) Tokens",
        f"{prompt_tokens:,}",
        f"{stats['total_prompt_tokens']:,}"
    )
    table.add_row(
        "Completion (Output) Tokens",
        f"{completion_tokens:,}",
        f"{stats['total_completion_tokens']:,}"
    )
    table.add_row(
        "Total Tokens",
        f"{prompt_tokens + completion_tokens:,}",
        f"{stats['total_tokens']:,}"
    )
    table.add_row(
        "Estimated Cost (USD)",
        f"${step_cost:.6f}",
        f"${stats['total_cost_usd']:.6f}"
    )
    
    console.print(Panel(
        table,
        border_style="dim",
        expand=False
    ))


def main() -> None:
    """Wrapper that starts the asyncio loop."""
    try:
        asyncio.run(run_chat_loop())
    except KeyboardInterrupt:
        console.print("\n[bold magenta]👋 Goodbye! Thanks for watching Build Sunday![/bold magenta]")
    except Exception as e:
        console.print(f"[bold red]❌ Event loop failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
