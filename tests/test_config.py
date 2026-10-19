import pytest

from api.clients import MagpieClient
from api.config import configs, get_model_config, iterate_files, resolve_provider


def make_repo(root):
    (root / "README.md").write_text("")
    (root / "CHANGELOG.md").write_text("")
    (root / "yarn.lock").write_text("") # excluded

    folder = root / "folder"
    folder.mkdir(exist_ok=True)
    (folder / ".lock").write_text("")   # excluded
    (folder / "code.py").write_text("")

    ex_folder = root / ".venv"
    ex_folder.mkdir(exist_ok=True)
    (ex_folder / "file.txt").write_text("")
    (ex_folder / ".gitignore").write_text("")



def test_iterate_files_default_exclusive_mode(exclude_test_config, tmp_path):
    make_repo(tmp_path)

    files = set(iterate_files(root_dir=str(tmp_path)))
    assert files == {
        "README.md",
        "CHANGELOG.md",
        "folder/code.py",
    }

@pytest.mark.parametrize(
    "included_dirs",
    [
        ["folder"],
        ["./folder"],
    ]
)
def test_iterate_files_included_dirs(exclude_test_config, tmp_path, included_dirs):
    make_repo(tmp_path)
    files = set(iterate_files(root_dir=str(tmp_path), included_dirs=included_dirs))
    assert files == {"folder/code.py"}


def test_iterate_files_included_files(exclude_test_config, tmp_path):
    make_repo(tmp_path)
    files = set(iterate_files(root_dir=str(tmp_path), included_files=["README.md"]))
    assert files == {"README.md"}


@pytest.mark.parametrize(
    "excluded_dirs",
    [
        ["folder"],
        ["./folder"],
    ]
)
def test_iterate_files_excluded_dirs(exclude_test_config, tmp_path, excluded_dirs):
    make_repo(tmp_path)
    files = set(iterate_files(root_dir=str(tmp_path), excluded_dirs=excluded_dirs))
    assert files == {"README.md", "CHANGELOG.md"}


# --------------------------------------------------------------------------- #
# Magpie provider configuration resolution
# --------------------------------------------------------------------------- #
def test_magpie_provider_resolves_from_environment(monkeypatch):
    from api.config import load_generator_config

    monkeypatch.setenv("MAGPIE_MODEL", "group/env-group")

    magpie = load_generator_config()["providers"]["magpie"]

    # Default model is taken from the environment placeholder...
    assert magpie["default_model"] == "group/env-group"
    # ...custom models are disallowed and the client class is wired up...
    assert magpie["supportsCustomModel"] is False
    assert magpie["client_class"] == "MagpieClient"
    assert magpie["model_client"] is MagpieClient
    # ...and there is exactly one Routing Group entry.
    assert len(magpie["models"]) == 1


def test_get_model_config_magpie_unknown_routing_group_does_not_raise():
    # The operator changed MAGPIE_MODEL to a value not listed in the models
    # map; resolving it must not raise and must carry that model through.
    result = get_model_config("magpie", "group/unlisted-group")

    assert result["model_client"] is MagpieClient
    assert result["model_kwargs"]["model"] == "group/unlisted-group"


def test_get_model_config_magpie_uses_configured_default_model():
    default_model = configs["providers"]["magpie"]["default_model"]

    result = get_model_config("magpie", None)

    assert result["model_client"] is MagpieClient
    assert result["model_kwargs"]["model"] == default_model


def test_models_config_endpoint_exposes_magpie_as_single_group_provider():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/models/config")
        assert response.status_code == 200
        providers = {p["id"]: p for p in response.json()["providers"]}

    magpie = providers["magpie"]
    assert magpie["supportsCustomModel"] is False
    assert len(magpie["models"]) == 1
    # Regression: the other providers are still present and unchanged.
    assert {"google", "openai", "openrouter", "ollama"} <= set(providers)
    assert response.json()["defaultProvider"] == "google"


# --------------------------------------------------------------------------- #
# Locked Provider: the single coercion decision
# --------------------------------------------------------------------------- #
def test_lock_resolver_returns_locked_provider_when_request_omits_provider(monkeypatch):
    # The request schema defaults the provider to "google" and the frontend
    # drops empty values, so an omitted Provider arrives as "" (or the default).
    # Either way the lock must win.
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")

    assert resolve_provider("") == "magpie"
    assert resolve_provider(None) == "magpie"


def test_lock_resolver_returns_locked_provider_when_request_names_another_provider(
    monkeypatch,
):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")

    assert resolve_provider("google") == "magpie"
    assert resolve_provider("openai") == "magpie"


def test_lock_resolver_returns_request_provider_when_lock_unset(monkeypatch):
    monkeypatch.delenv("DEEPWIKI_LOCK_PROVIDER", raising=False)

    assert resolve_provider("openai") == "openai"
    assert resolve_provider("google") == "google"


@pytest.mark.parametrize("blank", ["", "   "])
def test_lock_resolver_treats_blank_lock_as_unset(monkeypatch, blank):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", blank)

    assert resolve_provider("openai") == "openai"


# --------------------------------------------------------------------------- #
# Locked Provider exposed through the model-configuration response
# --------------------------------------------------------------------------- #
def _models_config():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/models/config")
    assert response.status_code == 200
    return response.json()


def test_models_config_reports_locked_provider_and_its_routing_group(monkeypatch):
    monkeypatch.setenv("DEEPWIKI_LOCK_PROVIDER", "magpie")

    body = _models_config()

    assert body["lockedProvider"] == "magpie"
    # The locked model is the locked provider's single Routing Group.
    assert body["lockedModel"] == configs["providers"]["magpie"]["default_model"]
    # The lock is additive: the full provider list is still returned.
    assert body["defaultProvider"] == "google"
    assert {"google", "openai", "magpie"} <= {p["id"] for p in body["providers"]}


def test_models_config_is_unchanged_when_lock_unset(monkeypatch):
    monkeypatch.delenv("DEEPWIKI_LOCK_PROVIDER", raising=False)

    body = _models_config()

    # Regression: with no lock the response carries no extra fields at all.
    assert set(body) == {"providers", "defaultProvider"}
    assert body["defaultProvider"] == "google"
