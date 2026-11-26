"""Provider-reported token usage for one LLM call.

A character-based token estimate is about 58% of the real count on Hungarian
text, so cost rows and the daily budget guard read the provider's own numbers
whenever a response carries them.

Two pieces live here, and deliberately nothing that touches the database, because
`lib/llm.py` imports this module and sits below every feature:

- `LlmUsage` and `usage_from_metadata`: the numbers Gemini bills, read from a
  response's `usage_metadata`.
- a context variable holding the last non-streamed call's usage. The attempt
  loggers (`recommender.llm_attempt_log` for the parser, rerank and region hint)
  run in the same task right after
  `await client.complete(...)`, so they can log real tokens
  without widening the `LlmClient` Protocol that every mock implements.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """One call's billed tokens, as the provider reported them.

    `prompt_tokens` INCLUDES `cached_tokens`: Gemini's `prompt_token_count` is the
    whole effective prompt, and `cached_content_token_count` says how much of it
    was served from a cache (implicit or explicit) at the cached rate. Thinking
    tokens are billed as output but reported separately, so they stay separate.
    """

    model: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def billed_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens

    def with_model(self, model: str) -> LlmUsage:
        return replace(self, model=model)


def _count(metadata: object, name: str) -> int:
    value = getattr(metadata, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def usage_from_metadata(metadata: object, *, model: str = "") -> LlmUsage | None:
    """`usage_metadata` → `LlmUsage`; None when the response carried no counts.

    A cached count larger than the prompt would be a provider bug, but it would
    make the uncached share negative and the cost too low, so it is clamped.
    """
    if metadata is None:
        return None
    prompt = _count(metadata, "prompt_token_count")
    output = _count(metadata, "candidates_token_count")
    thinking = _count(metadata, "thoughts_token_count")
    if prompt == 0 and output == 0 and thinking == 0:
        return None
    return LlmUsage(
        model=model,
        prompt_tokens=prompt,
        cached_tokens=min(_count(metadata, "cached_content_token_count"), prompt),
        output_tokens=output,
        thinking_tokens=thinking,
    )


_LAST_USAGE: ContextVar[LlmUsage | None] = ContextVar("ingovo_last_llm_usage", default=None)


def reset_last_llm_usage() -> None:
    """Forget the previous call's usage. Call sites do this right before a call, so
    a failed or mocked call can never be logged with an earlier call's numbers."""
    _LAST_USAGE.set(None)


def set_last_llm_usage(usage: LlmUsage | None) -> None:
    _LAST_USAGE.set(usage)


def last_llm_usage() -> LlmUsage | None:
    """What the last `GoogleAiClient.complete` in this task reported, if it succeeded."""
    return _LAST_USAGE.get()


__all__ = [
    "LlmUsage",
    "last_llm_usage",
    "reset_last_llm_usage",
    "set_last_llm_usage",
    "usage_from_metadata",
]
