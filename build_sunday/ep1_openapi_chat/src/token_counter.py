"""
Token Counter & Cost Estimator Module.

This module is responsible for:
1. Counting tokens in text strings and structured chat message payloads.
2. Estimating costs based on the model's pricing.
3. Keeping a cumulative session tracker for tokens and cost.
4. Explaining WHAT tokens are and HOW tokenization works in LLMs.

WHAT ARE TOKENS? (Educational Insight):
--------------------------------------
1. Tokenization: LLMs do not read text character-by-character or word-by-word.
   Instead, they break text into chunks of characters called "tokens". A token can be
   a single letter, a syllable, a whole word (e.g. "the"), or part of a word (e.g. "ing").
2. Rule of Thumb: In English, 1 token is roughly 4 characters, and 100 tokens correspond to
   about 75 words.
3. Tiktoken: OpenAI uses a byte-pair encoding (BPE) tokenizer. We use `tiktoken` to run the
   exact same tokenizer locally, allowing us to compute token usage before sending requests
   and validating our costs.
"""

import tiktoken
from typing import Dict, List, Any


# Pricing per 1,000,000 tokens (USD)
# Prices updated to OpenAI rates:
# - gpt-4o-mini: $0.15 / 1M prompt tokens, $0.60 / 1M completion tokens
# - gpt-4o: $2.50 / 1M prompt tokens, $10.00 / 1M completion tokens
# - gpt-3.5-turbo: $0.50 / 1M prompt tokens, $1.50 / 1M completion tokens
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt_cost_per_1m": 0.150,
        "completion_cost_per_1m": 0.600
    },
    "gpt-4o": {
        "prompt_cost_per_1m": 2.500,
        "completion_cost_per_1m": 10.000
    },
    "gpt-3.5-turbo": {
        "prompt_cost_per_1m": 0.500,
        "completion_cost_per_1m": 1.500
    }
}

# Fallback pricing if the model is unrecognized (e.g., custom deployment)
DEFAULT_PRICING = {
    "prompt_cost_per_1m": 0.500,
    "completion_cost_per_1m": 1.500
}


class TokenTracker:
    """
    Tracks token consumption and session expenses.
    
    Provides utility methods to count tokens for strings and message dictionaries,
    and maintains running totals for session token usage and financial cost.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost: float = 0.0
        
        # Initialize the tokenizer for the specified model
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to cl100k_base which is used by GPT-4 and GPT-3.5 models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_string_tokens(self, text: str) -> int:
        """Count tokens in a single raw string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in structured message dictionaries exactly as OpenAI Chat Completions does.
        
        Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        
        Each message has a role, content, and an optional name. The API packages these into a specific
        format that introduces a fixed token overhead per message.
        """
        num_tokens = 0
        
        # Message overhead determines formatting tokens.
        # For modern chat models (gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo-0125, etc.):
        # Every message is formatted as:
        # <|start|>{role/name}\n{content}<|end|>\n
        # This incurs a 3 token overhead per message.
        # If a name is present, the role is omitted from formatting, but the name takes a 1 token penalty.
        tokens_per_message = 3
        tokens_per_name = 1

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        # Every response is primed with <|start|>assistant\n which counts as 3 tokens.
        num_tokens += 3
        return num_tokens

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for a single operation in USD.
        
        Formula: (tokens / 1,000,000) * cost_per_1m
        """
        # Determine pricing based on the configured model
        pricing = MODEL_PRICING.get(self.model_name, DEFAULT_PRICING)
        
        prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt_cost_per_1m"]
        completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion_cost_per_1m"]
        
        return prompt_cost + completion_cost

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Record usage for a response, update the session statistics, and return the step's cost.
        """
        step_cost = self.calculate_cost(prompt_tokens, completion_tokens)
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += step_cost
        
        return step_cost

    def get_session_stats(self) -> Dict[str, Any]:
        """Return a dictionary of the session's cumulative metrics."""
        return {
            "model_name": self.model_name,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost
        }
