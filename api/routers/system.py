from datetime import datetime

from fastapi import APIRouter

from api.config import (
    LockedProviderMisconfiguredError,
    configs,
    get_locked_model,
    get_locked_provider,
)
from api.logger import get_logger
from api.schemas import Model, ModelConfig, Provider

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api",
    }


@router.get("/lang/config")
async def lang_config():
    return configs["lang_config"]


@router.get(
    "/models/config", response_model=ModelConfig, response_model_exclude_none=True
)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    While the deployment is locked (DEEPWIKI_LOCK_PROVIDER set) the response
    additionally reports the locked provider and its Routing Group; the lock
    fields are omitted entirely when unlocked so the response is unchanged.
    A locked deployment whose Routing Group environment variable is unset fails
    loudly (LockedProviderMisconfiguredError) instead of reporting the raw
    placeholder.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # Add models from config
            for model_id in provider_config["models"].keys():
                # Get a more user-friendly display name if possible
                models.append(Model(id=model_id, name=model_id))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get(
                        "supportsCustomModel", False
                    ),
                    models=models,
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider,
            **locked_provider_fields(),
        )
        return config

    except LockedProviderMisconfiguredError:
        # A misconfigured lock is fatal, not something to paper over with the
        # fallback configuration: surface it (and its logged message).
        raise
    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash")],
                )
            ],
            defaultProvider="google",
            **locked_provider_fields(),
        )


def locked_provider_fields() -> dict:
    """The additive lock fields for the model-configuration response.

    Empty when the deployment is unlocked, so the response keeps its exact
    previous shape; otherwise the locked provider id and its Routing Group.
    """
    locked_provider = get_locked_provider()
    if not locked_provider:
        return {}
    return {
        "lockedProvider": locked_provider,
        "lockedModel": get_locked_model(),
    }
