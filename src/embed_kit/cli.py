import asyncio
import sys
from pathlib import Path

import click

from embed_kit.models.adapters.registry import get_adapter_registry
from embed_kit.utils.config import Settings
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.cli")


def validate_port(ctx, param, value):
    if value < 1 or value > 65535:
        raise click.BadParameter(f"Port must be between 1 and 65535, got {value}")
    return value


def validate_config_file(ctx, param, value):
    config_path = Path(value)
    if not config_path.exists():
        raise click.BadParameter(f"Configuration file not found: {value}")
    if not config_path.is_file():
        raise click.BadParameter(f"Configuration path is not a file: {value}")
    
    try:
        Settings.from_yaml(config_path)
    except Exception as e:
        raise click.BadParameter(f"Invalid configuration file: {e}")
    
    return value


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    pass


@cli.command()
@click.option("--host", required=True, help="Host to bind to")
@click.option("--port", required=True, type=int, callback=validate_port, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
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
    reload: bool,
    log_level: str,
    model: str,
    config: str,
) -> None:
    import uvicorn
    
    settings = Settings.from_yaml(config)
    
    APP_NAME = settings.app.name
    APP_VERSION = settings.app.version
    DEFAULT_LOG_LEVEL = settings.app.default_log_level
    
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Configuration: host={host}, port={port}, model={model}")
    logger.debug(f"Config file: {config}")
    
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level_upper = log_level.upper()
    if log_level_upper not in valid_log_levels:
        logger.warning(f"Invalid log level '{log_level}', using '{DEFAULT_LOG_LEVEL}'")
        log_level_upper = DEFAULT_LOG_LEVEL
    
    try:
        from embed_kit.main import create_app
        app = create_app(
            app_name=APP_NAME,
            app_version=APP_VERSION,
            selected_model=model,
            config_path=config,
        )
        
        uvicorn.run(
            app,
            host=host,
            port=port,
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
            settings = Settings.from_yaml(config)
        except Exception as e:
            logger.error(f"Failed to read configuration file: {e}")
            click.echo(f"Error: Failed to read configuration file: {e}", err=True)
            return False
        
        try:
            model_config = settings.get_model_config(model)
        except ValueError as e:
            logger.error(str(e))
            click.echo(f"Error: {e}", err=True)
            return False
        
        model_params = {
            "model_path": model_config.path,
            "model_name": model_config.get_model_name(),
            "use_fp16": model_config.use_fp16,
            "device": model_config.device,
            "normalize_embeddings": model_config.normalize_embeddings,
            "batch_size": model_config.batch_size,
            "max_length": model_config.max_length,
        }
        
        logger.debug(f"Loading model with config: {model_params}")
        click.echo(f"Loading model: {model}")
        
        handler = None
        try:
            from embed_kit.models.handler import create_model_handler
            handler = create_model_handler(model, **model_params)
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
            from embed_kit.models.handler import EmbeddingInput
            
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
    help="Path to models config file (optional, shows models if provided)",
)
def info(config: str | None = None) -> None:
    """Display system information: registered handlers and configured models"""
    adapters = list(get_adapter_registry().keys())
    click.echo("=" * 70)
    click.echo("EmbedKit System Information")
    click.echo("=" * 70)
    
    click.echo("\n📋 Registered Adapters:")
    if not adapters:
        click.echo("  No adapters registered")
    else:
        for adapter_name in adapters:
            click.echo(f"  • {adapter_name}")
    
    if config:
        try:
            config_path = Path(config)
            if not config_path.exists():
                click.echo(f"\n⚠️  Config file not found: {config}")
                return
            
            settings = Settings.from_yaml(config)
            
            click.echo(f"\n📦 Configured Models ({config}):")
            if not settings.models:
                click.echo("  No models found in configuration")
            else:
                for model_id, model_config in settings.models.items():
                    model_name = model_config.name or "N/A"
                    model_path = model_config.path
                    
                    status = "✅"
                    click.echo(f"\n  {status} {model_id}")
                    click.echo(f"     Name: {model_name}")
                    click.echo(f"     Path: {model_path}")
        except Exception as e:
            logger.error(f"Failed to read config: {e}")
            click.echo(f"\n❌ Error reading config: {e}")
    
    click.echo("\n" + "=" * 70)


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
        settings = Settings.from_yaml(config)
        
        click.echo(f"\nFound {len(settings.models)} model(s):")
        for model_id, model_config in settings.models.items():
            model_name = model_config.name or "N/A"
            click.echo(f"  ✅ {model_id} (name: {model_name})")
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
