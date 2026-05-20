from pydantic import BaseModel, Field


class EmbeddingInput(BaseModel):
    texts: list[str]
    model: str | None = None
    batch_size: int = Field(default=32, gt=0)
    max_length: int = Field(default=512, gt=0)


class DenseEmbeddingOutput(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    num_texts: int


class SparseEmbedding(BaseModel):
    indices: list[int]
    values: list[float]


class MultiEmbeddingOutput(BaseModel):
    dense_embeddings: list[list[float]]
    sparse_embeddings: list[SparseEmbedding] | None = None
    model: str
    dimensions: int
    num_texts: int
