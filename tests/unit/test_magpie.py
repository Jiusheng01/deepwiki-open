"""Magpie client wiring: its own credentials and base URL, never OpenAI's."""

from api.chat import ChatStreamer
from api.clients import MagpieClient


def test_magpie_client_reads_its_own_base_url_and_key(monkeypatch):
    monkeypatch.setenv("MAGPIE_BASE_URL", "http://magpie.local:9999")
    monkeypatch.setenv("MAGPIE_API_KEY", "mp-key")
    # The native OpenAI key must NOT be consulted by Magpie.
    monkeypatch.setenv("OPENAI_API_KEY", "native-openai-key")

    client = MagpieClient()

    assert client.base_url == "http://magpie.local:9999/v1"
    assert client._env_api_key_name == "MAGPIE_API_KEY"


def test_magpie_client_uses_its_own_default_base_url(monkeypatch):
    monkeypatch.delenv("MAGPIE_BASE_URL", raising=False)

    client = MagpieClient()

    assert client.base_url == "http://localhost:8080/v1"


def test_create_magpie_streamer_builds_magpie_client(monkeypatch):
    # Exercises the real construction path end to end (no mocks): the provider
    # registry resolves `magpie` to a streamer wired to a MagpieClient.
    monkeypatch.setenv("MAGPIE_BASE_URL", "http://magpie.local:9999")
    monkeypatch.setenv("MAGPIE_API_KEY", "mp-key")

    streamer = ChatStreamer.create(
        provider="magpie",
        model="group/example",
        model_config={"model": "group/example", "temperature": 0.7},
    )

    assert isinstance(streamer, ChatStreamer)
    assert isinstance(streamer.client, MagpieClient)
    assert streamer.client.base_url == "http://magpie.local:9999/v1"
