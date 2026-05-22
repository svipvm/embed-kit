from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from embed_kit.api.routes import set_model_handler
from embed_kit.utils.config import Settings
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.main")


class AppConfig(BaseModel):
    app_name: str = Field(..., description="Application name")
    app_version: str = Field(..., description="Application version")
    selected_model: str = Field(..., description="Selected model name")
    config_path: str = Field(..., description="Configuration file path")


def _load_selected_model(config: AppConfig) -> Any:
    logger.info(f"Loading model: {config.selected_model} from {config.config_path}")
    
    settings = Settings.from_yaml(config.config_path)
    model_config = settings.get_model_config(config.selected_model)
    
    model_params = {
        "model_path": model_config.path,
        "model_name": model_config.get_model_name(),
        "use_fp16": model_config.use_fp16,
        "device": model_config.device,
        "normalize_embeddings": model_config.normalize_embeddings,
        "batch_size": model_config.batch_size,
        "max_length": model_config.max_length,
    }
    
    handler_type = config.selected_model
    
    logger.debug(f"Model config: {model_params}")
    
    try:
        from embed_kit.models.handler import create_model_handler
        handler = create_model_handler(handler_type, **model_params)
        logger.info(f"Model handler created: {handler_type}")
        return handler
    except Exception as e:
        raise RuntimeError(f"Failed to create model handler '{handler_type}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application starting up")
    
    config = getattr(app.state, "config", None)
    if not config:
        raise ValueError("Application config not found in app.state")
    
    logger.info(f"Application config: {config.model_dump()}")
    
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


def create_app(
    app_name: str = "EmbedKit",
    app_version: str = "0.1.0",
    selected_model: str | None = None,
    config_path: str | None = None,
) -> FastAPI:
    if not selected_model:
        raise ValueError(
            "Model must be specified via command-line argument: "
            "embed-kit serve --model <model_name>"
        )
    if not config_path:
        raise ValueError(
            "Config file must be specified via command-line argument: "
            "embed-kit serve --config <config_path>"
        )
    
    config = AppConfig(
        app_name=app_name,
        app_version=app_version,
        selected_model=selected_model,
        config_path=config_path,
    )
    
    logger.info(f"Creating FastAPI app: {config.app_name} v{config.app_version}")
    
    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    app.state.config = config
    
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
            model_id = getattr(handler, "model_id", handler.config.get("model_name", "unknown"))
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


__all__ = ["create_app"]
