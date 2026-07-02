"""Test utilities for Traceroot SDK tests."""

from uuid import uuid4


def create_uuid() -> str:
    """Create a unique identifier for tests."""
    return str(uuid4())


def reset_traceroot() -> None:
    """Reset Traceroot global state between tests."""
    import traceroot
    from traceroot.instrumentation._instrumentors import AutogenInstrumentor, PydanticAIInstrumentor
    from traceroot.instrumentation.livekit import LiveKitInstrumentor

    if traceroot.get_client():
        traceroot.shutdown()
    traceroot._client = None
    AutogenInstrumentor._instrumented = False
    PydanticAIInstrumentor._instrumented = False
    LiveKitInstrumentor._instrumented = False
    import traceroot.instrumentation.livekit as livekit_instrumentation

    livekit_instrumentation._LIVEKIT_PROVIDER = None
    livekit_instrumentation._HIJACK_WARNING_EMITTED = False
