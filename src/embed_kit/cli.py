import asyncio
import os
import sys
from pathlib import Path

import click
import yaml

from embed_kit.models.registry import registry
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.cli")

APP_NAME = "EmbedKit"
APP_VERSION = "0.1.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_WORKERS = 1
DEFAULT_MODELS_CONFIG_PATH = "config/models.yaml"
DEFAULT_LOG_LEVEL = "INFO"


def validate_port(ctx, param, value):
    if value < 1 or value > 65535:
        raise click.BadParameter(f"Port must be between 1 and 65535, got {value}")
    return value


def validate_workers(ctx, param, value):
    if value < 1:
        raise click.BadParameter(f"Workers must be at least 1, got {value}")
    return value


def validate_config_file(ctx, param, value):
    config_path = Path(value)
    if not config_path.exists():
        raise click.BadParameter(f"Configuration file not found: {value}")
    if not config_path.is_file():
        raise click.BadParameter(f"Configuration path is not a file: {value}")
    
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                raise click.BadParameter(f"Configuration file is empty: {value}")
            if "models" not in config_data:
                raise click.BadParameter(f"Configuration file missing 'models' section: {value}")
    except yaml.YAMLError as e:
        raise click.BadParameter(f"Invalid YAML format in configuration file: {e}")
    
    return value


@click.group()
@click.version_option(version=APP_VERSION)
def cli() -> None:
    pass


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Host to bind to")
@click.option("--port", default=DEFAULT_PORT, type=int, callback=validate_port, help="Port to bind to")
@click.option("--workers", default=DEFAULT_WORKERS, type=int, callback=validate_workers, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
@click.option("--log-level", default=DEFAULT_LOG_LEVEL, help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
@click.option(
    "--model",
    required=True,
    help="Model to use (e.g., bge-m3)",
)
@click.option(
    "--config",
    required=True,
    callback=validate_config_file,
    help="Path to models config file",
)
def serve(
    host: str,
    port: int,
    workers: int,
    reload: bool,
    log_level: str,
    model: str,
    config: str,
) -> None:
    import uvicorn
    
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Configuration: host={host}, port={port}, workers={workers}, model={model}")
    logger.debug(f"Config file: {config}")
    
    if reload and workers > 1:
        logger.warning("Auto-reload is enabled, workers will be set to 1")
        workers = 1
    
    os.environ["EMBED_KIT_APP_NAME"] = APP_NAME
    os.environ["EMBED_KIT_APP_VERSION"] = APP_VERSION
    os.environ["EMBED_KIT_HOST"] = host
    os.environ["EMBED_KIT_PORT"] = str(port)
    os.environ["EMBED_KIT_WORKERS"] = str(workers)
    os.environ["EMBED_KIT_SELECTED_MODEL"] = model
    os.environ["EMBED_KIT_MODELS_CONFIG_PATH"] = config
    
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level_upper = log_level.upper()
    if log_level_upper not in valid_log_levels:
        logger.warning(f"Invalid log level '{log_level}', using '{DEFAULT_LOG_LEVEL}'")
        log_level_upper = DEFAULT_LOG_LEVEL
    
    try:
        uvicorn.run(
            "embed_kit.main:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level_upper.lower(),
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


@cli.command()
@click.option("--model", required=True, help="Model to test")
@click.option(
    "--config",
    required=True,
    callback=validate_config_file,
    help="Path to models config file",
)
def test_model(model: str, config: str) -> None:
    async def _test() -> bool:
        logger.info(f"Testing model: {model}")
        
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f) or {}
                models_config = config_data.get("models", {})
        except Exception as e:
            logger.error(f"Failed to read configuration file: {e}")
            click.echo(f"Error: Failed to read configuration file: {e}", err=True)
            return False
        
        if model not in models_config:
            available_models = list(models_config.keys())
            logger.error(f"Model '{model}' not found in config")
            click.echo(f"Error: Model '{model}' not found in config", err=True)
            click.echo(f"Available models: {available_models}", err=True)
            return False
        
        model_config = models_config[model]
        handler_type = model_config.get("type")
        
        if not handler_type:
            logger.error(f"Model '{model}' has no 'type' field")
            click.echo(f"Error: Model '{model}' has no 'type' field", err=True)
            return False
        
        logger.debug(f"Loading model with config: {model_config.get('config', {})}")
        click.echo(f"Loading model: {model}")
        
        handler = None
        try:
            handler = registry.create(handler_type, **model_config.get("config", {}))
            await handler.initialize()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            click.echo(f"Error: Failed to load model: {e}", err=True)
            if handler:
                try:
                    await handler.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error during shutdown: {shutdown_error}")
            return False
        
        logger.info("Model loaded successfully")
        click.echo("✅ Model loaded successfully!")
        
        try:
            from embed_kit.models.embedding import EmbeddingInput
            
            test_input = EmbeddingInput(texts=["Hello, world!", "Test embedding"])
            result = await handler.process(test_input)
            
            logger.debug(f"Test result: dimensions={result.dimensions}, num_texts={result.num_texts}")
            click.echo(f"✅ Embedding dimension: {result.dimensions}")
            click.echo(f"✅ Number of embeddings: {result.num_texts}")
            click.echo(f"✅ First embedding (first 5 values): {result.embeddings[0][:5]}")
            
            await handler.shutdown()
            logger.info("Model test complete")
            return True
        except Exception as e:
            logger.error(f"Failed to test model: {e}")
            click.echo(f"Error: Failed to test model: {e}", err=True)
            try:
                await handler.shutdown()
            except Exception as shutdown_error:
                logger.error(f"Error during shutdown: {shutdown_error}")
            return False
    
    success = asyncio.run(_test())
    if not success:
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    required=True,
    callback=validate_config_file,
    help="Path to models config file",
)
def list_models(config: str) -> None:
    try:
        with open(config) as f:
            config_data = yaml.safe_load(f) or {}
            models_config = config_data.get("models", {})
        
        click.echo("Available models:")
        if not models_config:
            click.echo("  No models found in configuration")
            return
        
        for model_name, model_config in models_config.items():
            handler_type = model_config.get("type", "unknown")
            model_params = model_config.get("config", {})
            model_name_in_config = model_params.get("model_name", "N/A")
            
            click.echo(f"\n  📦 {model_name}")
            click.echo(f"     Type: {handler_type}")
            click.echo(f"     Model: {model_name_in_config}")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        click.echo(f"Error: Failed to list models: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_handlers() -> None:
    handlers = registry.list_handlers()
    click.echo("Registered handlers:")
    if not handlers:
        click.echo("  No handlers registered")
        return
    
    for handler_name in handlers:
        click.echo(f"  - {handler_name}")


@cli.command()
@click.option(
    "--config",
    required=True,
    callback=validate_config_file,
    help="Path to models config file",
)
def validate_config(config: str) -> None:
    click.echo(f"✅ Configuration file is valid: {config}")
    
    try:
        with open(config) as f:
            config_data = yaml.safe_load(f) or {}
            models_config = config_data.get("models", {})
        
        click.echo(f"\nFound {len(models_config)} model(s):")
        for model_name, model_config in models_config.items():
            handler_type = model_config.get("type", "unknown")
            model_params = model_config.get("config", {})
            
            required_params = ["model_name", "model_path", "batch_size", "max_length"]
            missing_params = [p for p in required_params if p not in model_params]
            
            if missing_params:
                click.echo(f"  ❌ {model_name} (type: {handler_type})")
                click.echo(f"     Missing required parameters: {missing_params}")
            else:
                click.echo(f"  ✅ {model_name} (type: {handler_type})")
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        click.echo(f"Error: Failed to validate configuration: {e}", err=True)
        sys.exit(1)


def main() -> None:
    try:
        cli()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"Error: Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
