"""HTTP contract: while the deployment is locked, a generation request that
omits its Provider is served by the locked Provider, not by the schema's
hard-coded "google" default.

Ask, Deep Research and Wiki generation all reach the model through the chat
route (wiki tasks reuse ``research_chat``); CodeMap has its own route. Both are
covered here. Only the LLM network call (``respond_stream``) and the retriever
are replaced — everything from the route down through the provider registry
runs for real.
"""

import json

from fastapi.testclient import TestClient

import api.services.codemap as codemap
import api.services.research as research
from api.chat._stream import (
    GoogleGenerativeChatStreamer,
    MagpieChatStreamer,
    OpenAIChatStreamer,
)


class _FakeRAG:
    """Stands in for the prepared retriever (no index / embeddings involved)."""

    def memory(self) -> dict:
        return {}

    async def acall(self, query, language="en"):
        return []


def _stub_llm(monkeypatch):
    """Replace the model network calls with markers naming their provider."""

    def marker(name):
        async def respond_stream(self, prompt):
            yield f"answered-by:{name}"

        return respond_stream

    for streamer, name in (
        (MagpieChatStreamer, "magpie"),
        (GoogleGenerativeChatStreamer, "google"),
        (OpenAIChatStreamer, "openai"),
    ):
        monkeypatch.setattr(streamer, "respond_stream", marker(name))


async def _stub_prepare_repo_index(request):
    return _FakeRAG()


def _stream(body: dict) -> str:
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/chat/completions/stream", json=body)
    assert response.status_code == 200
    return response.text


_NO_PROVIDER = {
    "repo_url": "https://github.com/o/r",
    # No "provider" key at all: the schema would default it to "google" and
    # the frontend drops empty selectors.
    "messages": [{"role": "user", "content": "hi"}],
}


def test_provider_less_chat_request_is_served_by_locked_provider(monkeypatch):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")
    monkeypatch.setattr(research, "prepare_repo_index", _stub_prepare_repo_index)
    _stub_llm(monkeypatch)

    text = _stream(_NO_PROVIDER)

    assert "answered-by:magpie" in text
    assert "answered-by:google" not in text


def test_chat_request_naming_another_provider_is_coerced_to_lock(monkeypatch):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")
    monkeypatch.setattr(research, "prepare_repo_index", _stub_prepare_repo_index)
    _stub_llm(monkeypatch)

    text = _stream({**_NO_PROVIDER, "provider": "openai"})

    assert "answered-by:magpie" in text
    assert "answered-by:openai" not in text


def test_chat_routing_unchanged_when_lock_unset(monkeypatch):
    monkeypatch.delenv("DEEPWIKI_LOCK_PROVIDER", raising=False)
    monkeypatch.setattr(research, "prepare_repo_index", _stub_prepare_repo_index)
    _stub_llm(monkeypatch)

    # Regression: unlocked, a provider-less request still resolves to the
    # schema default exactly as before this change.
    assert "answered-by:google" in _stream(_NO_PROVIDER)
    assert "answered-by:openai" in _stream({**_NO_PROVIDER, "provider": "openai"})


# --------------------------------------------------------------------------- #
# CodeMap — the fourth generation path (its own streamer call site)
# --------------------------------------------------------------------------- #
def _stub_codemap_llm(monkeypatch):
    """Each provider's streamer returns a minimal codemap naming itself."""

    def codemap_responder(fake_provider):
        async def respond_stream(self, prompt):
            yield json.dumps({"title": fake_provider, "sections": []})

        return respond_stream

    for streamer, name in (
        (MagpieChatStreamer, "magpie"),
        (GoogleGenerativeChatStreamer, "google"),
    ):
        monkeypatch.setattr(streamer, "respond_stream", codemap_responder(name))

    class _FakeCodeMapRAG:
        db_manager = None  # skip citation grounding against real sources

        def __init__(self, *args, **kwargs):
            pass

        async def aprepare_retriever(self, *args, **kwargs):
            return None

        async def acall(self, *args, **kwargs):
            return []

    monkeypatch.setattr(codemap, "RAG", _FakeCodeMapRAG)


def test_codemap_generation_is_coerced_to_locked_provider(monkeypatch):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")
    _stub_codemap_llm(monkeypatch)

    from api.main import app

    with TestClient(app) as client:
        response = client.post(
            "/codemap/stream",
            json={"repo_url": "https://github.com/o/r", "question": "how?"},
        )

    assert response.status_code == 200
    assert '"title": "magpie"' in response.text
