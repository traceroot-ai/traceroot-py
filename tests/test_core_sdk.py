"""Core SDK tests - initialization, singleton, shutdown."""

import traceroot
from tests.utils import reset_traceroot


def test_initialize_returns_client():
    """Test initialize() returns a client instance."""
    reset_traceroot()
    client = traceroot.initialize(api_key="test-key", enabled=False)

    assert client is not None
    assert traceroot.get_client() is client


def test_disabled_without_api_key():
    """Test client is disabled when no API key provided."""
    reset_traceroot()
    client = traceroot.initialize()

    assert client.enabled is False


def test_reinitialize_is_noop():
    """Test re-initializing returns the same client without replacing it."""
    reset_traceroot()

    client1 = traceroot.initialize(api_key="key1", enabled=False)
    client2 = traceroot.initialize(api_key="key2", enabled=False)

    assert client2 is client1
    assert traceroot.get_client() is client1


def test_shutdown():
    """Test shutdown() marks client as not initialized."""
    reset_traceroot()

    traceroot.initialize(api_key="test-key", enabled=False)
    traceroot.shutdown()

    assert traceroot.get_client()._initialized is False

def test_flush_without_client_is_safe():
    """Test flush() does nothing and does not raise when no client exists."""
    reset_traceroot()
    traceroot._client = None

    # Should not raise
    traceroot.flush()


def test_flush_with_client_does_not_raise():
    """Test flush() calls through to the client without raising."""
    reset_traceroot()
    traceroot.initialize(api_key="test-key", enabled=False)

    # Disabled clients have no span_processor, so flush is a no-op.
    # This locks in that flush() never raises regardless of client state.
    traceroot.flush()


def test_get_client_auto_initializes():
    """Test get_client() creates a client when none exists."""
    reset_traceroot()
    traceroot._client = None

    client = traceroot.get_client()

    assert client is not None
    assert traceroot.get_client() is client


def test_get_client_returns_existing_client():
    """Test get_client() returns the existing client without replacing it."""
    reset_traceroot()
    client = traceroot.initialize(api_key="test-key", enabled=False)

    assert traceroot.get_client() is client
