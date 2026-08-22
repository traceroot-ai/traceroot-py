"""Span processor for Traceroot OpenTelemetry integration.

This module defines the TracerootSpanProcessor class, which extends
OpenTelemetry's BatchSpanProcessor with Traceroot-specific configuration.
"""

import logging
import os
import threading
from collections import OrderedDict

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    Compression,
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from traceroot.constants import (
    DEFAULT_FLUSH_AT,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_TIMEOUT,
    SDK_NAME,
    SDK_VERSION,
)
from traceroot.env import (
    TRACEROOT_FLUSH_AT,
    TRACEROOT_FLUSH_INTERVAL,
    TRACEROOT_TIMEOUT,
)

logger = logging.getLogger(__name__)

_PATH_MAP_MAX: int = 1024
_PathMap = OrderedDict[str, list[str]]


class TracerootSpanProcessor(BatchSpanProcessor):
    """OpenTelemetry span processor that exports spans to Traceroot API.

    This processor extends OpenTelemetry's BatchSpanProcessor with
    Traceroot-specific
    configuration and defaults. It uses the standard OTLPSpanExporter to send
    OTLP-formatted trace data (protobuf) to the Traceroot backend.

    The API layer handles protobuf → JSON conversion before storing to S3.

    Features:
    - Configurable batch size and flush interval via constructor or env vars
    - Automatic batching and periodic flushing
    - Graceful shutdown with final flush
    - OTLP HTTP-based span export with gzip compression
    """

    def __init__(
        self,
        *,
        api_key: str,
        host_url: str,
        flush_at: int | None = None,
        flush_interval: float | None = None,
        timeout: float | None = None,
        git_repo: str | None = None,
        git_ref: str | None = None,
    ):
        """Initialize the span processor.

        Args:
            api_key: Traceroot API key for authentication.
            host_url: Traceroot API host URL.
            flush_at: Max batch size before flush. Falls back to
                TRACEROOT_FLUSH_AT
                env var, then DEFAULT_FLUSH_AT.
            flush_interval: Seconds between automatic flushes. Falls back to
                TRACEROOT_FLUSH_INTERVAL env var, then DEFAULT_FLUSH_INTERVAL.
            timeout: HTTP request timeout in seconds. Falls back to
                TRACEROOT_TIMEOUT env var, then DEFAULT_TIMEOUT.
            git_repo: Repository in "owner/repo" format. When provided, stamped
                as ``traceroot.git.repo`` on every recording span.
            git_ref: Git reference (commit SHA, tag, branch). When provided,
                stamped as ``traceroot.git.ref`` on every recording span.
        """
        # Resolve flush_at with env var fallback
        if flush_at is None:
            env_flush_at = os.environ.get(TRACEROOT_FLUSH_AT)
            flush_at = int(env_flush_at) if env_flush_at else DEFAULT_FLUSH_AT

        # Resolve flush_interval with env var fallback
        if flush_interval is None:
            env_flush_interval = os.environ.get(TRACEROOT_FLUSH_INTERVAL)
            flush_interval = (
                float(env_flush_interval) if env_flush_interval else DEFAULT_FLUSH_INTERVAL
            )

        # Resolve timeout with env var fallback
        if timeout is None:
            env_timeout = os.environ.get(TRACEROOT_TIMEOUT)
            timeout = float(env_timeout) if env_timeout else DEFAULT_TIMEOUT

        # Build endpoint URL
        endpoint = f"{host_url.rstrip('/')}/api/v1/public/traces"

        # Create the standard OTLP exporter (protobuf format)
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "x-traceroot-sdk-name": SDK_NAME,
                "x-traceroot-sdk-version": SDK_VERSION,
            },
            timeout=int(timeout),
            compression=Compression.Gzip,
        )

        # Initialize parent BatchSpanProcessor
        super().__init__(
            span_exporter=exporter,
            max_export_batch_size=flush_at,
            schedule_delay_millis=int(flush_interval * 1000),
        )

        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._git_repo = git_repo
        self._git_ref = git_ref
        self._paths_lock = threading.RLock()
        # Bounded OrderedDict: evicts oldest entry when capacity is exceeded,
        # eliminating the on_end race where a parent was removed before a
        # concurrent sibling's on_start could look it up.
        self._ids_path_by_span_id: _PathMap = OrderedDict()
        self._name_path_by_span_id: _PathMap = OrderedDict()
        # Unlike ids_path, each map value includes the mapped span's own start
        # time so a child can inherit the complete root-to-parent chain.
        self._starts_path_by_span_id: _PathMap = OrderedDict()

    def on_start(self, span, parent_context=None):
        if span.is_recording():
            span.set_attribute("traceroot.sdk.name", SDK_NAME)
            span.set_attribute("traceroot.sdk.version", SDK_VERSION)
            if self._git_repo:
                span.set_attribute("traceroot.git.repo", self._git_repo)
            if self._git_ref:
                span.set_attribute("traceroot.git.ref", self._git_ref)

            try:
                # span.parent is the SpanContext of the parent (set by the SDK at
                # creation time from the active context). It is always correct even
                # when the parent is a remote/NonRecordingSpan with no attributes.
                parent_ctx = getattr(span, "parent", None)
                parent_id_hex = (
                    format(parent_ctx.span_id, "016x")
                    if parent_ctx and parent_ctx.is_valid
                    else None
                )

                with self._paths_lock:
                    # Prefer the in-process map: OpenInference creates LangGraph node
                    # spans with a remote/NonRecordingSpan parent that carries no
                    # attributes, so reading parent_span.attributes would give None
                    # and break the ancestry chain.
                    parent_ids_path: list | None = (
                        self._ids_path_by_span_id.get(parent_id_hex) if parent_id_hex else None
                    )
                    parent_path: list | None = (
                        self._name_path_by_span_id.get(parent_id_hex) if parent_id_hex else None
                    )
                    parent_starts_path: list | None = (
                        self._starts_path_by_span_id.get(parent_id_hex) if parent_id_hex else None
                    )

                    # Fall back to reading from the active parent span's attributes.
                    if parent_ids_path is None or parent_path is None or parent_starts_path is None:
                        if parent_context is not None:
                            parent_span = otel_trace.get_current_span(parent_context)
                        else:
                            parent_span = otel_trace.get_current_span()

                        attrs = getattr(parent_span, "attributes", None)
                        if attrs is not None:
                            raw_path = attrs.get("traceroot.span.path")
                            if raw_path is not None:
                                parent_path = list(raw_path)
                            raw_ids = attrs.get("traceroot.span.ids_path")
                            if raw_ids is not None:
                                parent_ids_path = list(raw_ids)
                            raw_starts = attrs.get("traceroot.span.starts_path")
                            parent_start_time = getattr(parent_span, "start_time", None)
                            if raw_starts is not None and parent_start_time is not None:
                                parent_starts_path = [
                                    *list(raw_starts),
                                    str(parent_start_time),
                                ]

                    span_name = getattr(span, "name", "") or ""

                    # path: [root_name, ..., current_name]
                    span_path = (
                        (parent_path + [span_name]) if parent_path is not None else [span_name]
                    )

                    # ids_path: [root_id, ..., direct_parent_id]
                    if parent_id_hex:
                        span_ids_path = (
                            parent_ids_path + [parent_id_hex]
                            if parent_ids_path is not None
                            else [parent_id_hex]
                        )
                    else:
                        span_ids_path = []

                    # starts_path: ancestor epoch-nanosecond start times in the
                    # same root-to-parent order as ids_path. Never emit a partial
                    # or malformed chain because it could associate a timestamp
                    # with the wrong ancestor.
                    if parent_id_hex and parent_starts_path is not None:
                        candidate_starts_path = parent_starts_path
                    else:
                        candidate_starts_path = []

                    starts_path_is_valid = len(candidate_starts_path) == len(span_ids_path) and all(
                        isinstance(value, str) and value.isascii() and value.isdecimal()
                        for value in candidate_starts_path
                    )
                    span_starts_path = list(candidate_starts_path) if starts_path_is_valid else []

                    # Store in map so descendant spans can inherit via lookup.
                    # Evict the oldest entry if we're at capacity.
                    span_id_hex = format(span.context.span_id, "016x")
                    if len(self._ids_path_by_span_id) >= _PATH_MAP_MAX:
                        oldest_span_id, _ = self._ids_path_by_span_id.popitem(last=False)
                        self._name_path_by_span_id.pop(oldest_span_id, None)
                        self._starts_path_by_span_id.pop(oldest_span_id, None)
                    self._ids_path_by_span_id[span_id_hex] = span_ids_path
                    self._name_path_by_span_id[span_id_hex] = span_path
                    span_start_time = getattr(span, "start_time", None)
                    self._starts_path_by_span_id[span_id_hex] = (
                        [*span_starts_path, str(span_start_time)]
                        if starts_path_is_valid and span_start_time is not None
                        else []
                    )

                span.set_attribute("traceroot.span.path", span_path)
                span.set_attribute("traceroot.span.ids_path", span_ids_path)
                span.set_attribute("traceroot.span.starts_path", span_starts_path)

            except Exception as exc:
                logger.debug("TracerootSpanProcessor: failed to set path attributes: %s", exc)

        super().on_start(span, parent_context)

    def on_end(self, span):
        with self._paths_lock:
            span_id_hex = format(span.context.span_id, "016x")
            self._ids_path_by_span_id.pop(span_id_hex, None)
            self._name_path_by_span_id.pop(span_id_hex, None)
            self._starts_path_by_span_id.pop(span_id_hex, None)
        super().on_end(span)

    @property
    def flush_at(self) -> int:
        """Get the configured batch size."""
        return self._flush_at

    @property
    def flush_interval(self) -> float:
        """Get the configured flush interval in seconds."""
        return self._flush_interval
