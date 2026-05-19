from typing import Any, Callable, Type

from embed_kit.models.base import BaseModelHandler
from embed_kit.utils.logger import get_logger

logger = get_logger("embed_kit.models.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Type[BaseModelHandler]] = {}

    def register(self, name: str, handler_class: Type[BaseModelHandler]) -> None:
        if name in self._handlers:
            logger.warning(f"Handler '{name}' already registered, overwriting")
        self._handlers[name] = handler_class
        logger.debug(f"Registered handler: {name} -> {handler_class.__name__}")

    def register_handler(
        self, name: str
    ) -> Callable[[Type[BaseModelHandler]], Type[BaseModelHandler]]:
        def decorator(cls: Type[BaseModelHandler]) -> Type[BaseModelHandler]:
            self.register(name, cls)
            logger.info(f"Handler registered via decorator: {name} -> {cls.__name__}")
            return cls

        return decorator

    def get(self, name: str) -> Type[BaseModelHandler]:
        if name not in self._handlers:
            available = list(self._handlers.keys())
            raise ValueError(f"Unknown handler '{name}'. Available: {available}")
        return self._handlers[name]

    def list_handlers(self) -> list[str]:
        return list(self._handlers.keys())

    def create(self, name: str, **config: Any) -> BaseModelHandler:
        handler_class = self.get(name)
        logger.info(f"Creating handler '{name}' with config: {config}")
        return handler_class(**config)


registry = ModelRegistry()
