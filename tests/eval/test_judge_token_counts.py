"""``_default_complete`` surfaces the provider's token counts alongside the response text.

When no provider integration traces the judge model, the judge's own ``llm_judge:{name}`` span is
the ONLY record of that model call -- and the backend derives cost from the OpenInference
``llm.token_count.*`` attributes on ingest, so a span without them is a costless LLM call. Usage
used to be discarded here. The SDK never computes cost; it only reports what the provider returned.

The span-attribute half of this contract lives in ``test_trace_native.py::TestLlmJudgeTrace``.
"""

from __future__ import annotations

import sys
import types

from traceroot.eval import judge as judge_mod

MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _install_anthropic(monkeypatch, resp) -> None:
    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **kw: resp))
    monkeypatch.setitem(
        sys.modules, "anthropic", _fake_module("anthropic", Anthropic=lambda: client)
    )


def _install_openai(monkeypatch, resp) -> None:
    completions = types.SimpleNamespace(create=lambda **kw: resp)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    monkeypatch.setitem(sys.modules, "openai", _fake_module("openai", OpenAI=lambda: client))


class TestAnthropicUsage:
    def test_cache_tokens_fold_into_the_prompt_count(self, monkeypatch):
        _install_anthropic(
            monkeypatch,
            types.SimpleNamespace(
                content=[types.SimpleNamespace(text="0.8")],
                usage=_Usage(
                    input_tokens=20,
                    output_tokens=7,
                    cache_read_input_tokens=90,
                    cache_creation_input_tokens=10,
                ),
            ),
        )

        text, usage = judge_mod._default_complete("claude-sonnet-5", MESSAGES)

        assert text == "0.8"
        # Cache reads/writes ARE prompt tokens (billed at their own rate), so the prompt count
        # includes them -- the same arithmetic the anthropic auto-instrumentation uses, so the
        # judge's span and an auto-instrumented span report the same thing for the same call.
        assert usage == {
            "prompt": 120,
            "completion": 7,
            "total": 127,
            "cache_read": 90,
            "cache_creation": 10,
        }

    def test_uncached_call_reports_only_the_counts_it_has(self, monkeypatch):
        _install_anthropic(
            monkeypatch,
            types.SimpleNamespace(
                content=[types.SimpleNamespace(text="0.8")],
                usage=_Usage(input_tokens=20, output_tokens=7),
            ),
        )

        _, usage = judge_mod._default_complete("claude-sonnet-5", MESSAGES)

        assert usage == {"prompt": 20, "completion": 7, "total": 27}

    def test_response_without_usage_yields_none(self, monkeypatch):
        _install_anthropic(
            monkeypatch,
            types.SimpleNamespace(content=[types.SimpleNamespace(text="0.8")], usage=None),
        )

        assert judge_mod._default_complete("claude-sonnet-5", MESSAGES) == (
            "0.8",
            None,
        )


class TestOpenAIUsage:
    def test_usage_is_surfaced(self, monkeypatch):
        _install_openai(
            monkeypatch,
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.8"))],
                usage=_Usage(prompt_tokens=120, completion_tokens=7, total_tokens=127),
            ),
        )

        text, usage = judge_mod._default_complete("gpt-5", MESSAGES)

        assert text == "0.8"
        assert usage == {"prompt": 120, "completion": 7, "total": 127}

    def test_missing_total_is_derived(self, monkeypatch):
        _install_openai(
            monkeypatch,
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.8"))],
                usage=_Usage(prompt_tokens=120, completion_tokens=7),
            ),
        )

        _, usage = judge_mod._default_complete("gpt-5", MESSAGES)

        assert usage == {"prompt": 120, "completion": 7, "total": 127}

    def test_response_without_usage_yields_none(self, monkeypatch):
        _install_openai(
            monkeypatch,
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.8"))],
                usage=None,
            ),
        )

        assert judge_mod._default_complete("gpt-5", MESSAGES) == (
            "0.8",
            None,
        )
