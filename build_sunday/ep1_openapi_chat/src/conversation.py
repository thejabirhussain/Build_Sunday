"""
Conversation Memory Module for Build Sunday Q&A Bot.

This module is responsible for:
1. Storing the conversation history in memory.
2. Structuring user, assistant, and system roles for Chat Completions.
3. Truncating history dynamically when message tokens exceed limits (preventing out-of-context errors).
4. Explaining HOW Large Language Model memory works.

HOW DOES LLM MEMORY WORK? (Educational Insight):
----------------------------------------------
1. LLMs are completely STATELESS: They do not remember previous requests. Each API call is
   an isolated event. To simulate "memory" or "conversation state", we must package the entire
   transcript of previous turns (system prompts, user prompts, assistant replies) and send it
   back to the API with every new message.
2. Context Window: LLMs have a maximum limit of tokens they can read at once (the context window).
   If we keep appending messages infinitely, we will eventually breach the context limit and 
   receive an API error.
3. Memory Strategies:
   - Rolling Window: Truncate old messages to keep the token size within bounds.
   - Summarization: Summarize older portions of the chat to compress them (advanced).
   Here, we implement the Rolling Window approach: keeping the system prompt intact, and
   discarding the oldest user/assistant messages if the token budget is exceeded.
"""

from typing import List, Dict, Optional
from src.token_counter import TokenTracker


class ConversationHistory:
    """
    Manages the running conversation history and format structures for the API.
    
    Provides methods to add messages, output format lists, and truncate older context
    based on token limits while preserving the critical system prompt.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, concise, and smart Q&A assistant.",
        max_history_tokens: int = 4096,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the conversation history.
        
        Args:
            system_prompt: The base instruction that sets the behavior/personality of the assistant.
            max_history_tokens: The maximum allowable tokens in the history before truncation begins.
            token_tracker: An optional TokenTracker instance to compute exact token lengths.
        """
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        
        # If no tracker is provided, instantiate a default one for length checking
        self.tracker = token_tracker if token_tracker is not None else TokenTracker("gpt-4o-mini")
        
        # Internal list of message dictionaries: [{"role": "user", "content": "..."}]
        self._messages: List[Dict[str, str]] = []
        
        # Always inject the system prompt as the first message
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        if content.strip():
            self._messages.append({"role": "user", "content": content.strip()})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to the conversation history."""
        if content.strip():
            self._messages.append({"role": "assistant", "content": content.strip()})

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Return the list of messages in the format expected by OpenAI Chat Completions API.
        
        Performs truncation if the total history length exceeds the token budget.
        """
        self._truncate_if_needed()
        return self._messages

    def clear(self) -> None:
        """Reset the conversation history, keeping only the system prompt."""
        self._messages = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

    def _truncate_if_needed(self) -> None:
        """
        Trims the conversation history to keep it within the max_history_tokens threshold.
        
        We MUST preserve:
        1. The system prompt (usually at index 0).
        2. The most recent messages.
        
        We discard:
        - The oldest user/assistant pairs.
        """
        while len(self._messages) > 1:
            total_tokens = self.tracker.count_messages_tokens(self._messages)
            
            # If we are within budget, we stop truncating
            if total_tokens <= self.max_history_tokens:
                break
                
            # If the only message left after system prompt is the current user message,
            # we cannot truncate further without losing the active question.
            if len(self._messages) <= 2:
                break
                
            # Remove the oldest exchange (index 1 is the oldest message after the system prompt)
            # We pop index 1 twice to remove both the old User query and Assistant response pair.
            # If for some reason the structure is off, we just pop once to avoid infinite loop.
            removed_role = self._messages[1].get("role")
            self._messages.pop(1)
            
            # If we popped a user message, try to also pop the matching assistant response
            # that follows it immediately to keep context pairs clean.
            if removed_role == "user" and len(self._messages) > 1 and self._messages[1].get("role") == "assistant":
                self._messages.pop(1)
