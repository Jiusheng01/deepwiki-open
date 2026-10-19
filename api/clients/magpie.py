import os
from typing import Optional

from adalflow.components.model_client.openai_client import OpenAIClient

from .litellm import LiteLLMClient


class MagpieClient(LiteLLMClient):
    """
    Magpie OpenAI-compatible gateway client.

    Magpie speaks the OpenAI API, so we reuse the existing OpenAI-compatible
    gateway client (``LiteLLMClient``) — inheriting its HTTP client
    construction and parsing — and override only the environment variable
    names and the default base URL. No new HTTP request logic is written here.

    Credentials are read from Magpie's own environment variables — never from
    the native OpenAI configuration — so enabling Magpie cannot disturb an
    existing OpenAI setup.

    Expected environment variables:

    MAGPIE_BASE_URL=http://localhost:8080
    MAGPIE_API_KEY=...
    """

    # A different default from the LiteLLM gateway's port (4000).
    DEFAULT_BASE_URL = "http://localhost:8080"

    def __init__(
        self,
        api_key: Optional[str] = None,
        input_type: str = "text",
        base_url: Optional[str] = None,
    ):
        resolved_base_url = base_url or os.getenv(
            "MAGPIE_BASE_URL", self.DEFAULT_BASE_URL
        )
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url.rstrip('/')}/v1"

        # Initialise through adalflow's OpenAIClient directly: the installed
        # adalflow no longer accepts the ``chat_completion_parser`` /
        # ``env_base_url_name`` keyword arguments that ``LiteLLMClient.__init__``
        # forwards (so ``LiteLLMClient()`` itself cannot be constructed here).
        # We still inherit LiteLLMClient's ``init_sync_client`` /
        # ``init_async_client``, which build the OpenAI / AsyncOpenAI clients
        # from ``base_url`` and ``_env_api_key_name``.
        OpenAIClient.__init__(
            self,
            api_key=api_key,
            input_type=input_type,
            base_url=resolved_base_url,
            env_api_key_name="MAGPIE_API_KEY",
        )
