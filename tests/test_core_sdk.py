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


def test_environment_from_env_var(monkeypatch):
    """Test TRACEROOT_ENVIRONMENT is read when no environment argument is given."""
    reset_traceroot()
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "staging")

    client = traceroot.initialize(api_key="test-key", enabled=False)

    assert client.environment == "staging"


def test_environment_argument_overrides_env_var(monkeypatch):
    """Test the environment argument wins over TRACEROOT_ENVIRONMENT."""
    reset_traceroot()
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "staging")

    client = traceroot.initialize(api_key="test-key", enabled=False, environment="production")

    assert client.environment == "production"


def test_empty_environment_env_var_is_unset(monkeypatch):
    """Test an empty TRACEROOT_ENVIRONMENT is treated as unset."""
    reset_traceroot()
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "")

    client = traceroot.initialize(api_key="test-key", enabled=False)

    assert client.environment is None


def test_empty_environment_argument_overrides_env_var(monkeypatch):
    """Test environment="" turns off an inherited TRACEROOT_ENVIRONMENT."""
    reset_traceroot()
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "staging")

    client = traceroot.initialize(api_key="test-key", enabled=False, environment="")

    assert client.environment is None
