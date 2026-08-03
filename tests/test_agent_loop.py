"""P0: LLM agent loop must have a hard step cap so tool-call loops cannot run forever."""

import types

import pytest

from srun import llm as llm_module
from srun.context import state


class _FakeToolCall:
    def __init__(self, index):
        self.index = index
        self.id = f"call_{index}"
        self.function = types.SimpleNamespace(
            name="run_command",
            arguments='{"command": "echo hi", "language": "shell"}',
        )


class _FakeDelta:
    def __init__(self, tool_calls):
        self.content = None
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, tool_calls):
        self.delta = _FakeDelta(tool_calls)


class _FakeUsage:
    prompt_cache_miss_tokens = 0  # cache always hits → token budget never triggers
    prompt_cache_hit_tokens = 1000


class _FakeChunk:
    def __init__(self, tool_calls):
        self.usage = _FakeUsage()
        self.choices = [_FakeChoice(tool_calls)]


class _FakeCreate:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return iter([_FakeChunk([_FakeToolCall(self.calls)])])


class _FakeClient:
    def __init__(self):
        self.model = "fake-model"
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_FakeCreate())
        )


@pytest.fixture
def fake_llm(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(llm_module.llm, "client", client)
    monkeypatch.setattr(
        llm_module,
        "config_get",
        lambda key: 2 if key == "max_llm_steps" else 12,
    )
    state.reset_session()
    yield client
    monkeypatch.setattr(llm_module.llm, "client", None)


def _exec(cmd, lang):
    return True, "ok"


def test_agent_loop_stops_after_max_steps(fake_llm):
    """Model keeps issuing tool calls; loop must stop at max_llm_steps, not spin."""
    summary, tool_calls, conv = llm_module.llm.run(
        "do a long task", exec_callback=_exec
    )
    assert summary and "max steps" in summary
    assert tool_calls is not None and len(tool_calls) == 2
    assert fake_llm.chat.completions.create.calls == 2


def test_agent_loop_completes_within_step_budget(fake_llm, monkeypatch):
    """A model that answers in text after one tool call should terminate normally."""
    class _TextDelta:
        content = "all done"
        reasoning_content = None
        tool_calls = None

    class _TextChoice:
        delta = _TextDelta()

    class _TextChunk:
        usage = _FakeUsage()
        choices = [_TextChoice()]

    def _create(**kwargs):
        return iter([_TextChunk()])

    fake_llm.chat.completions.create = _create
    summary, tool_calls, conv = llm_module.llm.run(
        "list files", exec_callback=_exec
    )
    assert summary == "all done"
    assert tool_calls is None
