"""Traceroot client."""

import atexit
import logging
import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from traceroot.constants import (
    DEFAULT_FLUSH_AT,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_HOST_URL,
    DEFAULT_TIMEOUT,
)
from traceroot.env import (
    TRACEROOT_API_KEY,
    TRACEROOT_ENABLED,
    TRACEROOT_FLUSH_AT,
    TRACEROOT_FLUSH_INTERVAL,
    TRACEROOT_HOST_URL,
    TRACEROOT_TIMEOUT,
)
from traceroot.git_context import auto_detect_git_context, harvest_ci_git_context
from traceroot.instrumentation.registry import Integration
from traceroot.transport.span_processor import TracerootSpanProcessor

logger = logging.getLogger(__name__)

# Guard so the "git context unresolved" warning fires at most once per process.
_git_context_warning_lock = threading.Lock()
_git_context_warning_emitted: bool = False


def _warn_git_context_unresolved() -> None:
    """Emit a one-time warning when git context cannot be resolved from any source."""
    global _git_context_warning_emitted
    with _git_context_warning_lock:
        if _git_context_warning_emitted:
            return
        _git_context_warning_emitted = True
    logger.warning(
        "TraceRoot: Git context could not be resolved. "
        "Set TRACEROOT_GIT_REPO and TRACEROOT_GIT_REF in production so traces "
        "can be correlated to source code. "
        "See https://docs.traceroot.ai/tracing/git-context"
    )


class TracerootClient:
    """Main client for sending traces to Traceroot.

    The client initializes an OpenTelemetry TracerProvider
    with a span processor that exports OTLP-formatted trace
    data to the Traceroot backend.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host_url: str | None = None,
        flush_interval: float | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
        enabled: bool | None = None,
        integrations: list[Integration] | None = None,
        git_repo: str | None = None,
        git_ref: str | None = None,
    ):
        """Initialize the Traceroot client.

        Args:
            api_key: API key for authentication. Falls back
                to TRACEROOT_API_KEY env var.
            host_url: API host URL. Falls back to
                TRACEROOT_HOST_URL env var.
            flush_interval: Seconds between automatic flushes.
                Falls back to TRACEROOT_FLUSH_INTERVAL env
                var, then 5.0.
            batch_size: Maximum items per batch before flush.
                Falls back to TRACEROOT_FLUSH_AT env var,
                then 100.
            timeout: HTTP request timeout in seconds. Falls
                back to TRACEROOT_TIMEOUT env var, then 30.0.
            enabled: Whether tracing is enabled. Falls back
                to TRACEROOT_ENABLED env var.
            integrations: Libraries to auto-instrument
                (e.g. ["openai", "langchain"]).
            git_repo: Repository in "owner/repo" format.
                Falls back to TRACEROOT_GIT_REPO env var,
                CI/platform env vars (GitHub Actions, Vercel,
                GitLab CI, CircleCI, Bitbucket, Render),
                then auto-detected from git remote.
            git_ref: Git reference (commit SHA, tag, branch).
                Falls back to TRACEROOT_GIT_REF env var,
                CI/platform env vars, then auto-detected from
                git HEAD.
        """
        # Resolve config with env var fallbacks
        self.api_key = api_key or os.environ.get(TRACEROOT_API_KEY, "")
        self.host_url = host_url or os.environ.get(TRACEROOT_HOST_URL, DEFAULT_HOST_URL)

        if flush_interval is None:
            env_interval = os.environ.get(TRACEROOT_FLUSH_INTERVAL)
            flush_interval = float(env_interval) if env_interval else DEFAULT_FLUSH_INTERVAL
        self.flush_interval = flush_interval

        if batch_size is None:
            env_batch = os.environ.get(TRACEROOT_FLUSH_AT)
            batch_size = int(env_batch) if env_batch else DEFAULT_FLUSH_AT
        self.batch_size = batch_size

        if timeout is None:
            env_timeout = os.environ.get(TRACEROOT_TIMEOUT)
            timeout = float(env_timeout) if env_timeout else DEFAULT_TIMEOUT
        self.timeout = timeout

        if enabled is None:
            env_enabled = os.environ.get(TRACEROOT_ENABLED, "").lower()
            enabled = env_enabled not in ("false", "0", "no", "off") if env_enabled else True

        self._integrations = integrations


        _env_repo: str | None = os.environ.get("TRACEROOT_GIT_REPO") or None
        _env_ref: str | None = os.environ.get("TRACEROOT_GIT_REF") or None

        _ci_ctx: dict[str, str | None] = {}
        if not (git_repo or _env_repo) or not (git_ref or _env_ref):
            _ci_ctx = harvest_ci_git_context()
        _git_ctx: dict[str, str | None] = {}
        _need_repo = not (git_repo or _env_repo or _ci_ctx.get("git_repo"))
        _need_ref = not (git_ref or _env_ref or _ci_ctx.get("git_ref"))
        if _need_repo or _need_ref:
            _git_ctx = auto_detect_git_context()

        self.git_repo = (
            git_repo
            or _env_repo
            or _ci_ctx.get("git_repo")
            or _git_ctx.get("git_repo")
        )
        self.git_ref = (
            git_ref
            or _env_ref
            or _ci_ctx.get("git_ref")
            or _git_ctx.get("git_ref")
        )

        if not self.git_repo and not self.git_ref:
            _warn_git_context_unresolved()

        self._enabled = enabled and bool(self.api_key)
        if enabled and not self.api_key:
            logger.warning(
                "TraceRoot: no API key provided — tracing is disabled. "
                "Set the TRACEROOT_API_KEY environment variable or pass "
                "api_key= to traceroot.initialize()."
            )
        self._span_processor: TracerootSpanProcessor | None = None
        self._provider: TracerProvider | None = None
        self._initialized = False
        self._instrumented: list[Integration] = []

        if self._enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize TracerProvider with span processor."""
        if self._initialized:
            return

        # Create span processor
        self._span_processor = TracerootSpanProcessor(
            api_key=self.api_key,
            host_url=self.host_url,
            flush_at=self.batch_size,
            flush_interval=self.flush_interval,
            timeout=self.timeout,
        )

        # Create and configure TracerProvider
        self._provider = TracerProvider()
        self._provider.add_span_processor(self._span_processor)

        # Set as global provider so @observe decorator uses it
        trace.set_tracer_provider(self._provider)

        # Register shutdown handler
        atexit.register(self.shutdown)

        self._initialized = True
        logger.debug("Traceroot client initialized with TracerProvider")

        # Instrumentation (after TracerProvider is set up)
        if self._integrations is not None:
            from traceroot.instrumentation.registry import initialize_integrations

            self._instrumented = initialize_integrations(
                tracer_provider=self._provider,
                integrations=self._integrations,
            )

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    @property
    def span_processor(self) -> TracerootSpanProcessor | None:
        """Get the span processor for OTel integration."""
        return self._span_processor

    def flush(self) -> None:
        """Flush all pending traces."""
        if self._span_processor:
            self._span_processor.force_flush()

    def shutdown(self) -> None:
        """Shutdown the client gracefully."""
        if self._span_processor:
            self._span_processor.shutdown()
            self._span_processor = None

        self._provider = None
        self._initialized = False
        logger.debug("Traceroot client shutdown")
