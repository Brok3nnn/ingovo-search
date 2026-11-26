"""Shared type aliases and exception classes. Imports nothing application-specific."""

from ingovo_search.types.errors import (
    DbError,
    IngestionError,
    IngovoError,
    LlmBudgetExceededError,
    LlmError,
    ParseError,
    ValidationError,
)

__all__ = [
    "DbError",
    "IngestionError",
    "IngovoError",
    "LlmBudgetExceededError",
    "LlmError",
    "ParseError",
    "ValidationError",
]
