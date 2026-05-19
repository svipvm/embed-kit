from typing import Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]] = Field(
        ...,
        description="Input text to embed, can be a string or array of strings",
    )
    model: str = Field(
        ...,
        description="Model to use for embedding",
    )
    encoding_format: str = Field(
        ...,
        description="The format to return the embeddings in",
    )


class SparseEmbedding(BaseModel):
    indices: list[int] = Field(
        default_factory=list,
        description="Sparse token indices",
    )
    values: list[float] = Field(
        default_factory=list,
        description="Sparse token weights",
    )


class EmbeddingData(BaseModel):
    dense: list[float] = Field(
        default_factory=list,
        description="Dense embedding vector",
    )
    sparse: SparseEmbedding = Field(
        default_factory=SparseEmbedding,
        description="Sparse embedding",
    )


class EmbeddingObject(BaseModel):
    object: str
    embedding: EmbeddingData
    index: int


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str
    data: list[EmbeddingObject]
    model: str
    usage: UsageInfo
