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


def test_shutdown_resets_global_client_reference():
    """Test shutdown() clears the module-level client so it can't be reused."""
    reset_traceroot()

    traceroot.initialize(api_key="test-key", enabled=False)
    traceroot.shutdown()

    assert traceroot._client is None


def test_reinitialize_after_shutdown_creates_fresh_client():
    """Test initialize() after shutdown() returns a new client, not the dead one.

    Regression test for #117: shutdown() used to leave the dead client in
    place, so a later initialize() hit the singleton guard and silently
    returned the shutdown client instead of creating a fresh one.
    """
    reset_traceroot()

    client1 = traceroot.initialize(api_key="key1", enabled=False)
    traceroot.shutdown()
    client2 = traceroot.initialize(api_key="key2", enabled=False)

    assert client2 is not client1
    assert traceroot.get_client() is client2
