"""Application exception classes: one base (`IngovoError`) and one subclass per failure domain."""


class IngovoError(Exception):
    """Base for all application-level expected exceptions."""


class LlmError(IngovoError):
    """LLM call failed (transport, timeout, content violation)."""


class ParseError(LlmError):
    """LLM output failed validation against the expected schema."""


class LlmBudgetExceededError(LlmError):
    """Daily LLM spend cap exceeded for the requested model."""


class JevError(LlmError):
    """TypeSafe Jev call failed.

    `reason` is one of `http` (an unsuccessful status, see `status`), `timeout`,
    `connection`, `invalid_response` or `client` (SDK-side failure such as a
    malformed API key). `status` is the HTTP status when there was one.
    """

    def __init__(self, message: str, *, reason: str, status: int | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


class DbError(IngovoError):
    """Database access failed in an expected, recoverable way."""


class IngestionError(IngovoError):
    """Listing ingestion pipeline failed for an expected reason."""


class ValidationError(IngovoError):
    """Application-level validation (distinct from Pydantic field validation)."""


class RankingGuardrailViolation(IngovoError):
    """Ranking guardrail breached: paid placement in scoring, hard-filter
    exclusion broken by rerank, or a reranker movement beyond the allowed bound.
    See `recommender.ranking_guardrails`.
    """
