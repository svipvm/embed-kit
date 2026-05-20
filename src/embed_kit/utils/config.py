from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    name: str = Field(default="EmbedKit", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    default_log_level: str = Field(default="INFO", description="Default log level")

    @field_validator("default_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of {valid_levels}")
        return v_upper


class ModelConfig(BaseSettings):
    name: str | None = Field(default=None, description="Model name for loading")
    path: str = Field(..., description="Model file path")
    use_fp16: bool = Field(default=True, description="Use FP16 acceleration")
    device: str = Field(default="cuda", description="Device type (cuda/cpu)")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    batch_size: int = Field(..., description="Batch size for processing")
    max_length: int = Field(..., description="Maximum sequence length")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        return v

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_length must be at least 1")
        return v

    def get_model_name(self) -> str:
        if self.name:
            return self.name
        parts = self.path.rstrip("/").split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError("Configuration file is empty")
        
        return cls(**config_data)

    def get_model_config(self, model_id: str) -> ModelConfig:
        if model_id not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(
                f"Model '{model_id}' not found in config. Available models: {available_models}"
            )
        return self.models[model_id]

    def validate_model(self, model_id: str) -> None:
        model_config = self.get_model_config(model_id)
        required_params = ["path", "batch_size", "max_length"]
        missing_params = [p for p in required_params if not hasattr(model_config, p) or getattr(model_config, p) is None]
        if missing_params:
            raise ValueError(f"Missing required parameters for model '{model_id}': {missing_params}")
