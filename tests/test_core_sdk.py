"""Core SDK tests - initialization, singleton, shutdown."""

import pytest

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


@pytest.mark.asyncio
async def test_flush_async_calls_global_client_flush():
    """Test async flush wrapper works for async shutdown callbacks."""
    reset_traceroot()

    class Client:
        def __init__(self):
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1

        def shutdown(self):
            pass

    client = Client()
    traceroot._client = client

    await traceroot.flush_async()

    assert client.flush_count == 1


@pytest.mark.asyncio
async def test_flush_async_without_client_is_noop():
    """Test async flush wrapper is safe before initialization."""
    reset_traceroot()

    await traceroot.flush_async()
