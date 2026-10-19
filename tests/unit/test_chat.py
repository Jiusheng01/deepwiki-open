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
