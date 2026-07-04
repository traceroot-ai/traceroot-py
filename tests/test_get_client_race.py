"""Test for get_client() concurrent initialization race condition fix."""

import concurrent.futures
import unittest.mock

import pytest

import traceroot


@pytest.fixture(autouse=True)
def reset_client():
    """Reset global client before and after each test."""
    traceroot._client = None
    yield
    traceroot._client = None
    traceroot.shutdown()


def test_get_client_concurrent_single_instance():
    """Verify get_client() creates only one instance under concurrent load.

    This test ensures that the double-checked locking pattern in get_client()
    prevents the race condition where two concurrent calls both see _client=None
    and both attempt to instantiate TracerootClient.

    Regression test for issue #118.
    """
    # Mock the TracerootClient constructor to track instantiation count
    call_count = 0
    original_init = traceroot.TracerootClient.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Call original, but skip expensive initialization
        with unittest.mock.patch.object(self, "_initialize", return_value=None):
            original_init(self, *args, **kwargs)

    with (
        unittest.mock.patch.object(traceroot.TracerootClient, "__init__", counting_init),
        concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor,
    ):
        # Launch multiple threads calling get_client() concurrently
        futures = [executor.submit(traceroot.get_client) for _ in range(5)]
        results = [f.result() for f in futures]

    # All calls should return the same client instance
    assert all(client is results[0] for client in results), (
        "get_client() returned different instances across concurrent calls"
    )

    # Constructor should have been called exactly once, not five times
    assert call_count == 1, (
        f"TracerootClient.__init__ called {call_count} times, expected 1. "
        "Race condition: multiple concurrent get_client() calls created multiple instances."
    )


def test_get_client_idempotent_single_thread():
    """Verify get_client() returns the same instance when called repeatedly."""
    client1 = traceroot.get_client()
    client2 = traceroot.get_client()
    assert client1 is client2, "get_client() returned different instances on sequential calls"
