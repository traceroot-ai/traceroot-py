"""traceroot client."""

import atexit
import json
import logging
import os

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
    TRACEROOT_GIT_REF,
    TRACEROOT_GIT_REPO,
    TRACEROOT_GLOBAL_ATTRIBUTES,
    TRACEROOT_HOST_URL,
    TRACEROOT_TIMEOUT,
)
from traceroot.git_context import (
    auto_detect_git_context,
    git_context_from_files,
    harvest_ci_git_context,
)
from traceroot.instrumentation.registry import Integration
from traceroot.transport.span_processor import LocalEvalSampler, Redact, TracerootSpanProcessor

logger = logging.getLogger(__name__)

GlobalAttributeValue = str | int | float | bool


def _clean_global_attributes(raw: object) -> dict[str, GlobalAttributeValue] | None:
    """Keep only attributes OTel can carry on a span.

    Mirrors the TypeScript SDK's ``globalAttributes`` value type. Invalid
    entries are skipped with a warning instead of failing initialization:
    a malformed correlation tag must not stop the application from starting.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "traceroot: global attributes must be a dict of str/int/float/bool values; ignoring."
        )
        return None
    cleaned: dict[str, GlobalAttributeValue] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            logger.warning("traceroot: ignoring global attribute %r; unsupported key or value", key)
            continue
        cleaned[key] = value
    return cleaned or None


def _parse_global_attributes_env(value: str | None) -> dict[str, GlobalAttributeValue] | None:
    """Parse ``TRACEROOT_GLOBAL_ATTRIBUTES`` (a JSON object) at initialization."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        logger.warning("traceroot: ignoring %s; it is not valid JSON.", TRACEROOT_GLOBAL_ATTRIBUTES)
        return None
    return _clean_global_attributes(parsed)


class TracerootClient:
    """Main client for sending traces to traceroot.

    The client initializes an OpenTelemetry TracerProvider
    with a span processor that exports OTLP-formatted trace
    data to the traceroot backend.
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
        redact: Redact | None = None,
        global_attributes: dict[str, GlobalAttributeValue] | None = None,
    ):
        """Initialize the traceroot client.

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
                then auto-detected from git remote.
            git_ref: Git reference (commit SHA, tag, branch).
                Falls back to TRACEROOT_GIT_REF env var,
                then auto-detected from git HEAD.
            redact: Rewrites each finished span's attributes
                before export, so personal data never
                reaches traceroot (other exporters still see
                the original span). Also runs on event
                attributes and the status message.
                traceroot's own bookkeeping keys are not
                passed in; recorded input, output, metadata
                and prompt are. Eval results are redacted
                too; datasets and scorer definitions are
                not. A raising function fails closed: every
                value except numbers and booleans is
                withheld, and recorded metadata is dropped.
            global_attributes: Attributes stamped as span metadata on every
                span (root and children), so a value that a tool controls
                through the environment can be queried back by metadata.
                Falls back to the TRACEROOT_GLOBAL_ATTRIBUTES env var (a JSON
                object). Keys are used verbatim; reserve the ``traceroot.``
                prefix for SDK- and tool-set values. Per-call metadata wins
                on a key collision.
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
        self._redact = redact
        # Explicit wins over the env var entirely; an explicit empty dict means "none".
        if global_attributes is not None:
            self.global_attributes = _clean_global_attributes(global_attributes)
        else:
            self.global_attributes = _parse_global_attributes_env(
                os.environ.get(TRACEROOT_GLOBAL_ATTRIBUTES)
            )

        # Resolve git context: explicit > env > CI vars > auto-detect.
        # `or None` so an empty env var ("") is treated as unset — otherwise it
        # would block fallback detection and suppress the missing-context warning.
        self.git_repo = git_repo or os.environ.get(TRACEROOT_GIT_REPO) or None
        self.git_ref = git_ref or os.environ.get(TRACEROOT_GIT_REF) or None

        if self.git_repo is None or self.git_ref is None:
            ci = harvest_ci_git_context()
            self.git_repo = self.git_repo or ci["git_repo"]
            self.git_ref = self.git_ref or ci["git_ref"]

        if self.git_repo is None or self.git_ref is None:
            from_files = git_context_from_files()
            self.git_repo = self.git_repo or from_files["git_repo"]
            self.git_ref = self.git_ref or from_files["git_ref"]

        if self.git_repo is None or self.git_ref is None:
            auto = auto_detect_git_context()
            self.git_repo = self.git_repo or auto.get("git_repo")
            self.git_ref = self.git_ref or auto.get("git_ref")

        if self.git_repo is None or self.git_ref is None:
            logger.warning(
                "traceroot: git context incomplete (repo=%s, ref=%s). The AI "
                "agent needs both to correlate traces to source. Set "
                "TRACEROOT_GIT_REPO / TRACEROOT_GIT_REF — see "
                "https://traceroot.ai/docs/tracing/git-context",
                self.git_repo or "unset",
                self.git_ref or "unset",
            )

        self._enabled = enabled and bool(self.api_key)
        if enabled and not self.api_key:
            logger.warning(
                "traceroot: no API key provided — tracing is disabled. "
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
            git_repo=self.git_repo,
            git_ref=self.git_ref,
            redact=self._redact,
            global_attributes=self.global_attributes,
        )

        # OTel's global provider is set-once. If the host (e.g. `adk web`) already installed an SDK
        # provider, our set_tracer_provider() would be refused and @observe spans, which use the
        # global provider, would never reach our processor. Join it instead.
        existing = trace.get_tracer_provider()
        owns_provider = not isinstance(existing, TracerProvider)
        if not owns_provider:
            self._provider = existing
            logger.info(
                "traceroot: a TracerProvider is already installed globally; attaching the "
                "traceroot span processor to it instead of creating a new one."
            )
        else:
            self._provider = TracerProvider()
            if not isinstance(existing, trace.ProxyTracerProvider):
                logger.warning(
                    "traceroot: the global TracerProvider (%s) is not an OpenTelemetry SDK "
                    "TracerProvider, so it cannot be shared; @observe spans will not be sent "
                    "to traceroot.",
                    type(existing).__name__,
                )

        # Own provider: sample local=True eval spans DROP at creation. Joined provider: leave the
        # host's sampler (and so its own exporters) alone; TracerootSpanProcessor drops those spans.
        if owns_provider:
            self._provider.sampler = LocalEvalSampler(self._provider.sampler)
        self._provider.add_span_processor(self._span_processor)

        # Set as global provider so @observe decorator uses it
        if owns_provider:
            trace.set_tracer_provider(self._provider)

        # Register shutdown handler
        atexit.register(self.shutdown)

        self._initialized = True
        logger.debug("traceroot client initialized with TracerProvider")

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
        logger.debug("traceroot client shutdown")
