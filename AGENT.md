# EmbedKit 项目约束规范

## 核心架构

### 模型处理器接口
所有模型处理器必须继承 `BaseModelHandler[InputT, OutputT]`，实现：
- `input_schema` 和 `output_schema` 属性
- `async process(input_data) -> OutputT` 方法
- 通过装饰器 `@registry.register_handler("name")` 注册

### 配置管理
- 配置文件：`config/models.yaml`，必须包含 `model_name`, `model_path`, `batch_size`, `max_length`
- 环境变量：`EMBED_KIT_SELECTED_MODEL`, `EMBED_KIT_MODELS_CONFIG_PATH`, `EMBED_KIT_HOST`, `EMBED_KIT_PORT`
- `selected_model` 自动添加到配置中，供前端调用（如 `bge-m3`）
- `model_name` 为实际模型路径（如 `BAAI/bge-m3`）

### API 端点
OpenAI Embeddings API v1 兼容：
- `GET /health` - 健康检查，返回 `selected_model`
- `POST /v1/embeddings` - 创建 dense embeddings
- `POST /v1/embed_all` - 创建 dense + sparse embeddings
- `GET /v1/models` - 列出模型，返回 `selected_model` 作为 ID

### 请求格式
```json
{
  "input": "text or array",
  "model": "model-name",
  "encoding_format": "float"
}
```

### 响应格式
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": {"dense": [...], "sparse": {"indices": [...], "values": [...]}},
    "index": 0
  }],
  "model": "model-name",
  "usage": {"prompt_tokens": N, "total_tokens": N}
}
```

## CLI 命令
```bash
embed-kit serve --model bge-m3 --config config/models.yaml --port 8000
embed-kit test-model --model bge-m3 --config config/models.yaml
```

## 开发规范
- 所有配置项必须在 YAML 中明确指定，无代码默认值
- 新模型必须实现 `BaseModelHandler` 接口并注册
- 使用 Pydantic BaseModel 定义输入输出结构
- 异步处理所有 I/O 操作
