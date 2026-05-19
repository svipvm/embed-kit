from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseModelHandler(ABC, Generic[InputT, OutputT]):
    def __init__(self, **config: Any) -> None:
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            await self._setup()
            self._initialized = True

    async def _setup(self) -> None:
        pass

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[InputT]:
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[OutputT]:
        pass

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "handler": self.__class__.__name__,
            "initialized": self._initialized,
        }

    async def shutdown(self) -> None:
        self._initialized = False
