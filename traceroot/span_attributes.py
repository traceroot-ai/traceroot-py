"""Span attribute keys used by Traceroot SDK.

This module defines the OpenTelemetry span attribute keys used throughout
the SDK for consistent attribute naming.
"""


class SpanAttributes:
    """OTel span attribute keys used by Traceroot."""

    # =========================================================================
    # Span Attributes (core tracing)
    # =========================================================================
    SPAN_TYPE = "traceroot.span.type"
    SPAN_INPUT = "traceroot.span.input"
    SPAN_OUTPUT = "traceroot.span.output"
    SPAN_METADATA = "traceroot.span.metadata"
    SPAN_TAGS = "traceroot.span.tags"

    # =========================================================================
    # LLM-specific Attributes
    # =========================================================================
    LLM_MODEL = "traceroot.llm.model"
    LLM_MODEL_PARAMETERS = "traceroot.llm.model_parameters"
    LLM_USAGE = "traceroot.llm.usage"
    LLM_PROMPT = "traceroot.llm.prompt"

    # =========================================================================
    # Trace-level Attributes
    # =========================================================================
    TRACE_USER_ID = "traceroot.trace.user_id"
    TRACE_SESSION_ID = "traceroot.trace.session_id"
    TRACE_METADATA = "traceroot.trace.metadata"
    TRACE_TAGS = "traceroot.trace.tags"

    # =========================================================================
    # Git Context Attributes
    # =========================================================================
    GIT_REPO = "traceroot.git.repo"
    GIT_REF = "traceroot.git.ref"
    GIT_SOURCE_FILE = "traceroot.git.source_file"
    GIT_SOURCE_LINE = "traceroot.git.source_line"
    GIT_SOURCE_FUNCTION = "traceroot.git.source_function"

    # =========================================================================
    # System Attributes
    # =========================================================================
    ENVIRONMENT = "traceroot.environment"

    # =========================================================================
    # Offline-evaluation Attributes
    #
    # Additive, versioned identity contract for evaluation traces (see
    # offline-eval/contract-notes/eval-trace-attributes.md). EVAL_CONTRACT_VERSION
    # is bumped only on a breaking change; new keys are added, never repurposed.
    # =========================================================================
    EVAL_CONTRACT_VERSION = "traceroot.eval.contract_version"
    EVAL_NAME = "traceroot.eval.name"  # stable evaluation identity/purpose
    EVAL_RUN_NAME = "traceroot.eval.run_name"  # retained: run label (== name today)
    EVAL_RUN_ID = "traceroot.eval.run_id"  # platform run id when reported
    EVAL_DATASET_NAME = "traceroot.eval.dataset_name"
    EVAL_DATASET_ID = "traceroot.eval.dataset_id"
    EVAL_DATASET_VERSION_ID = "traceroot.eval.dataset_version_id"
    EVAL_CASE_ID = "traceroot.eval.case_id"
    EVAL_CANDIDATE_VERSION = "traceroot.eval.candidate_version"
    EVAL_ENVIRONMENT = "traceroot.eval.environment"
    EVAL_HAS_EXPECTED = "traceroot.eval.has_expected"
    EVAL_SOURCE_TRACE_ID = "traceroot.eval.source_trace_id"
    EVAL_SOURCE_SPAN_ID = "traceroot.eval.source_span_id"
    EVAL_SCORE_TARGET_SPAN_ID = "traceroot.eval.score_target_span_id"
    EVAL_TASK_NAME = "traceroot.eval.task_name"
    EVAL_ERROR = "traceroot.eval.error"
    EVAL_SCORER_NAME = "traceroot.eval.scorer_name"
    EVAL_SCORE_VALUE = "traceroot.eval.score_value"
    EVAL_SCORE_COMMENT = "traceroot.eval.score_comment"
