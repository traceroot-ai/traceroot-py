"""Traceroot Python SDK - Observability for LLM applications.

Basic Usage:
    import traceroot
    from traceroot import observe

    # Initialize (reads TRACEROOT_API_KEY from env)
    traceroot.initialize()

    # Use @observe decorator to trace functions
    @observe(name="my_agent", type="agent")
    def my_agent(query: str) -> str:
        return process(query)

    # Auto-instrument LangChain
    traceroot.initialize(integrations=[Integration.LANGCHAIN])

Session Management:
    from openinference.instrumentation import using_attributes

    with using_attributes(session_id="conv-123", user_id="user-456"):
        # All traces in this block get session_id and user_id
        response = client.chat.completions.create(...)
"""

import logging
import threading

# Re-export using_attributes from OpenInference for convenience
from openinference.instrumentation import using_attributes

from traceroot.client import TracerootClient
from traceroot.constants import SDK_VERSION as __version__
from traceroot.context import get_current_span_id, get_current_trace_id
from traceroot.decorators import observe
from traceroot.instrumentation import Integration
from traceroot.update import update_current_span, update_current_trace

logger = logging.getLogger(__name__)

# =============================================================================
# Global Singleton Client
# =============================================================================
_client: TracerootClient | None = None
_init_lock = threading.Lock()


def initialize(
    api_key: str | None = None,
    host_url: str | None = None,
    flush_interval: float | None = None,
    batch_size: int | None = None,
    timeout: float | None = None,
    enabled: bool | None = None,
    integrations: list[Integration] | None = None,
    git_repo: str | None = None,
    git_ref: str | None = None,
) -> TracerootClient:
    """Initialize the global Traceroot client.

    Call this once at application startup before using any tracing.

    Args:
        api_key: API key. Defaults to TRACEROOT_API_KEY env var.
        host_url: API host URL. Defaults to TRACEROOT_HOST_URL env var.
        flush_interval: Seconds between automatic flushes. Defaults to
            TRACEROOT_FLUSH_INTERVAL env var, then 5.0.
        batch_size: Max batch size before flush. Defaults to
            TRACEROOT_FLUSH_AT env var, then 100.
        timeout: HTTP request timeout in seconds. Defaults to
            TRACEROOT_TIMEOUT env var, then 30.0.
        enabled: Whether tracing is enabled. Defaults to
            TRACEROOT_ENABLED env var, then True.
        integrations: Libraries to auto-instrument. Use Integration enum
            values (e.g. [Integration.OPENAI, Integration.LANGCHAIN]).

    Returns:
        The TracerootClient instance.

    Example:
        import traceroot
        traceroot.initialize(
            integrations=[
                Integration.OPENAI,
                Integration.LANGCHAIN,
            ]
        )
    """
    global _client
    with _init_lock:
        if _client is not None:
            logger.warning(
                "traceroot.initialize() called more than once — ignoring. "
                "Call traceroot.shutdown() first if you need to reinitialize."
            )
            return _client
        _client = TracerootClient(
            api_key=api_key,
            host_url=host_url,
            flush_interval=flush_interval,
            batch_size=batch_size,
            timeout=timeout,
            enabled=enabled,
            integrations=integrations,
            git_repo=git_repo,
            git_ref=git_ref,
        )
    return _client


def get_client() -> TracerootClient | None:
    """Get the global Traceroot client, auto-initializing if needed.

    If no client exists and TRACEROOT_API_KEY is set, automatically
    initializes a client with default settings from environment variables.
    This enables the @observe decorator to work without explicit initialize().
    """
    global _client
    if _client is None:
        # Auto-initialize from env vars (TracerootClient reads them)
        _client = TracerootClient()
    return _client


def flush() -> None:
    """Flush all pending traces."""
    if _client:
        _client.flush()


def shutdown() -> None:
    """Shutdown the SDK gracefully."""
    if _client:
        _client.shutdown()


__all__ = [
    # Version
    "__version__",
    # Core
    "initialize",
    "get_client",
    "flush",
    "shutdown",
    "observe",
    "TracerootClient",
    "Integration",
    # Context utilities
    "get_current_trace_id",
    "get_current_span_id",
    # Update functions
    "update_current_span",
    "update_current_trace",
    # OpenInference (re-exported for convenience)
    "using_attributes",
    # Offline evaluation
    "Dataset",
    "DatasetSnapshot",
    "EvalCase",
    "Score",
    "ScorerContext",
    "DeferredScore",
    "Evaluation",
    "evaluate",
    "evaluate_async",
    "pull_dataset",
    "pull_dataset_version",
    "EvalRunResult",
    "EvalItemResult",
    "RunDatasetRef",
    "ScoreSummary",
    "UploadState",
]

# Offline-evaluation symbols are re-exported lazily so importing ``traceroot`` never
# eagerly pulls the eval submodules (keeps import light and safe on partial installs).
_EVAL_EXPORTS = frozenset(
    {
        "Dataset",
        "DatasetSnapshot",
        "DeferredScore",
        "EvalCase",
        "Score",
        "ScorerContext",
        "EvalItemResult",
        "EvalRunResult",
        "RunDatasetRef",
        "ScoreSummary",
        "UploadState",
        "Evaluation",
        "evaluate",
        "evaluate_async",
        "pull_dataset",
        "pull_dataset_version",
    }
)


def __getattr__(name):
    if name in _EVAL_EXPORTS:
        import traceroot.eval as _eval

        return getattr(_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
