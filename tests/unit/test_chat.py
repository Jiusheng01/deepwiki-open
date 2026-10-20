from types import SimpleNamespace

import pytest

from api.chat import ChatStreamer
from api.chat._stream import (
    AnthropicChatStreamer,
    AzureChatStreamer,
    BedrockChatStreamer,
    DashScopeChatStreamer,
    GoogleGenerativeChatStreamer,
    LiteLLMChatStreamer,
    MagpieChatStreamer,
    OllamaChatStreamer,
    OpenAIChatStreamer,
    OpenRouterChatStreamer,
)


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
        ("magpie", MagpieChatStreamer),
    ],
)
def test_every_provider_is_registered(provider, expected):
    assert ChatStreamer._registry[provider] is expected


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", OllamaChatStreamer),
        ("openrouter", OpenRouterChatStreamer),
        ("openai", OpenAIChatStreamer),
        ("azure", AzureChatStreamer),
        ("bedrock", BedrockChatStreamer),
        ("dashscope", DashScopeChatStreamer),
        ("google", GoogleGenerativeChatStreamer),
        ("litellm", LiteLLMChatStreamer),
        ("anthropic", AnthropicChatStreamer),
        ("magpie", MagpieChatStreamer),
    ],
)
def test_create_returns_correct_subclass(monkeypatch, provider, expected):
    monkeypatch.setattr(expected, "__init__", lambda self, **kw: None)
    s = ChatStreamer.create(provider=provider, model="m", model_config={"model": "m"})
    assert isinstance(s, expected)


def test_create_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="not registered"):
        ChatStreamer.create(provider="nope", model=None, model_config={})


# --------------------------------------------------------------------------- #
# Magpie: the configured Routing Group (MAGPIE_MODEL) is authoritative
# --------------------------------------------------------------------------- #
def _stub_magpie_client(monkeypatch):
    # Avoid constructing a real OpenAI-compatible client (and reading env) in
    # unit tests; we only assert on the kwargs Magpie would send.
    monkeypatch.setattr(MagpieChatStreamer, "_build_client", lambda self: object())


def test_magpie_streamer_uses_routing_group_from_env(monkeypatch):
    import api.chat._stream as stream_mod

    _stub_magpie_client(monkeypatch)
    monkeypatch.setattr(stream_mod, "MAGPIE_MODEL", "group/from-env")

    streamer = MagpieChatStreamer(
        model="whatever-the-request-carried", model_config={"temperature": 0.7}
    )

    assert streamer.model_kwargs["model"] == "group/from-env"


def test_magpie_streamer_tolerates_missing_model_config(monkeypatch):
    # A Routing Group not present in generator.json yields an empty parameter
    # set from get_model_config; constructing the streamer must not raise.
    import api.chat._stream as stream_mod

    _stub_magpie_client(monkeypatch)
    monkeypatch.setattr(stream_mod, "MAGPIE_MODEL", "group/unlisted")

    streamer = MagpieChatStreamer(model="m", model_config={"model": "m"})

    assert streamer.model_kwargs["model"] == "group/unlisted"
    assert "temperature" not in streamer.model_kwargs


def test_magpie_streamer_falls_back_to_request_model_without_env(monkeypatch):
    import api.chat._stream as stream_mod

    _stub_magpie_client(monkeypatch)
    monkeypatch.setattr(stream_mod, "MAGPIE_MODEL", None)

    streamer = MagpieChatStreamer(model="group/fallback", model_config={"temperature": 0.7})

    assert streamer.model_kwargs["model"] == "group/fallback"


# --------------------------------------------------------------------------- #
# Magpie streaming: the request goes straight to the gateway's chat completions
# endpoint. adalflow 1.1.3 routes OpenAI-compatible calls through the Responses
# API, whose events carry no `.choices`, so the shared base streamer cannot be
# used for Magpie.
# --------------------------------------------------------------------------- #
class _FakeAsyncStream:
    """Stands in for the OpenAI SDK's async chat-completions stream."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncClient:
    """Stands in for ``AsyncOpenAI`` at the SDK / network boundary."""

    def __init__(self, chunks):
        self.requests: list[dict] = []
        stream = _FakeAsyncStream(chunks)

        async def create(**kwargs):
            self.requests.append(kwargs)
            return stream

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


_EMPTY_CHUNK = SimpleNamespace(choices=[])


def _magpie_streamer(monkeypatch, async_client):
    # The async client is the external system boundary (the network), so it is
    # the one thing worth stubbing; everything else is real.
    monkeypatch.setattr(MagpieChatStreamer, "_build_client", lambda self: SimpleNamespace())
    monkeypatch.setattr(
        MagpieChatStreamer, "_build_async_client", lambda self: async_client
    )
    return MagpieChatStreamer(model="request-model", model_config={"temperature": 0.3})


async def _collect(streamer, prompt):
    return [text async for text in streamer.respond_stream(prompt)]


@pytest.mark.asyncio
async def test_magpie_respond_stream_yields_only_text_deltas(monkeypatch):
    client = _FakeAsyncClient([_chunk("PONG"), _EMPTY_CHUNK, _chunk(""), _chunk("!")])

    streamer = _magpie_streamer(monkeypatch, client)

    assert await _collect(streamer, "say PONG") == ["PONG", "!"]


@pytest.mark.asyncio
async def test_magpie_respond_stream_calls_chat_completions(monkeypatch):
    import api.chat._stream as stream_mod

    monkeypatch.setattr(stream_mod, "MAGPIE_MODEL", "group/ds")
    client = _FakeAsyncClient([_chunk("hi")])

    streamer = _magpie_streamer(monkeypatch, client)
    await _collect(streamer, "hello")

    (request,) = client.requests
    assert request["model"] == "group/ds"
    assert request["stream"] is True
    assert request["messages"] == [{"role": "user", "content": "hello"}]
