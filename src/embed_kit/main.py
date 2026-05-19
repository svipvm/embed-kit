import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from embed_kit.api.routes import set_model_handler
from embed_kit.models.registry import registry
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.main")


class AppConfig(BaseModel):
    app_name: str = Field(..., description="Application name")
    app_version: str = Field(..., description="Application version")
    selected_model: str = Field(..., description="Selected model name")
    config_path: str = Field(..., description="Configuration file path")


class ModelConfig(BaseModel):
    type: str = Field(..., description="Model handler type")
    config: dict[str, Any] = Field(default_factory=dict, description="Model configuration")


class ConfigValidation(BaseModel):
    models: dict[str, ModelConfig] = Field(..., description="Models configuration")


def _validate_config_file(config_path: str) -> ConfigValidation:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Configuration path is not a file: {config_path}")
    
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                raise ValueError("Configuration file is empty")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in configuration file: {e}")
    
    try:
        return ConfigValidation(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration structure: {e}")


def _get_app_config() -> AppConfig:
    app_name = os.environ.get("EMBED_KIT_APP_NAME")
    app_version = os.environ.get("EMBED_KIT_APP_VERSION")
    selected_model = os.environ.get("EMBED_KIT_SELECTED_MODEL")
    config_path = os.environ.get("EMBED_KIT_MODELS_CONFIG_PATH")
    
    missing_vars = []
    if not app_name:
        missing_vars.append("EMBED_KIT_APP_NAME")
    if not app_version:
        missing_vars.append("EMBED_KIT_APP_VERSION")
    if not selected_model:
        missing_vars.append("EMBED_KIT_SELECTED_MODEL")
    if not config_path:
        missing_vars.append("EMBED_KIT_MODELS_CONFIG_PATH")
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return AppConfig(
        app_name=app_name,
        app_version=app_version,
        selected_model=selected_model,
        config_path=config_path,
    )


def _load_selected_model(config: AppConfig) -> Any:
    logger.info(f"Loading model: {config.selected_model} from {config.config_path}")
    
    validated_config = _validate_config_file(config.config_path)
    
    if config.selected_model not in validated_config.models:
        available_models = list(validated_config.models.keys())
        raise ValueError(
            f"Model '{config.selected_model}' not found in config. "
            f"Available models: {available_models}"
        )
    
    model_config = validated_config.models[config.selected_model]
    handler_type = model_config.type
    model_params = model_config.config
    
    required_params = ["model_name", "model_path", "batch_size", "max_length"]
    missing_params = [p for p in required_params if p not in model_params]
    if missing_params:
        raise ValueError(
            f"Missing required parameters for model '{config.selected_model}': {missing_params}"
        )
    
    model_params["selected_model"] = config.selected_model
    
    logger.debug(f"Model config: {model_params}")
    
    try:
        handler = registry.create(handler_type, **model_params)
        logger.info(f"Model handler created: {handler_type}")
        return handler
    except Exception as e:
        raise RuntimeError(f"Failed to create model handler '{handler_type}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application starting up")
    
    try:
        config = _get_app_config()
        logger.info(f"Application config: {config.model_dump()}")
    except Exception as e:
        logger.error(f"Failed to load application config: {e}")
        raise
    
    handler = None
    try:
        handler = _load_selected_model(config)
        await handler.initialize()
        set_model_handler(handler)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        if handler:
            try:
                await handler.shutdown()
            except Exception as shutdown_error:
                logger.error(f"Error during shutdown: {shutdown_error}")
        raise
    
    shutdown_event = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_event
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        shutdown_event = True
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        yield
    finally:
        logger.info("Application shutting down")
        if handler:
            try:
                await handler.shutdown()
                logger.info("Model shutdown complete")
            except Exception as e:
                logger.error(f"Error during model shutdown: {e}")
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    try:
        config = _get_app_config()
    except Exception as e:
        logger.error(f"Failed to load application config: {e}")
        raise
    
    logger.info(f"Creating FastAPI app: {config.app_name} v{config.app_version}")
    
    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    from embed_kit.api.routes import router
    app.include_router(router, prefix="/v1")
    
    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "name": config.app_name,
            "version": config.app_version,
            "docs": "/docs",
            "redoc": "/redoc",
        }
    
    @app.get("/health")
    async def health() -> dict[str, Any]:
        from embed_kit.api.routes import get_model_handler
        try:
            handler = get_model_handler()
            model_id = handler.config.get("selected_model", handler.config.get("model_name", "unknown"))
            return {
                "status": "healthy",
                "model": model_id,
                "app": config.app_name,
                "version": config.app_version,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "app": config.app_name,
                "version": config.app_version,
            }
    
    @app.get("/ready")
    async def readiness() -> dict[str, Any]:
        from embed_kit.api.routes import get_model_handler
        try:
            handler = get_model_handler()
            health_status = await handler.health_check()
            return {
                "status": "ready",
                "health": health_status,
            }
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                "status": "not_ready",
                "error": str(e),
            }
    
    return app


app = create_app()

__all__ = ["app", "create_app"]
