# EmbedKit 项目约束规范

## 核心依赖
- **pydantic-settings**: 配置管理（>= 2.0.0）
- **fastapi**: Web 框架
- **click**: CLI 工具

## 核心架构

### 模型处理器接口
所有模型处理器必须继承 `BaseModelHandler[InputT, OutputT]`，实现：
- `input_schema` 和 `output_schema` 属性
- `async process(input_data) -> OutputT` 方法
- 通过装饰器 `@registry.register_handler("name")` 注册

### 配置管理
使用 **pydantic-settings** 管理配置（`config/settings.yaml`），配置类位于 `src/embed_kit/utils/config.py`：

```python
settings = Settings.from_yaml("config/settings.yaml")
model_config = settings.get_model_config("bge-m3")
```

**配置验证**：
- 必需字段：`path`, `batch_size`, `max_length`
- 自动验证：`batch_size >= 1`, `max_length >= 1`, 日志级别有效性
- `name` 可选，缺失时从 `path` 自动推断

**映射关系**：Model ID (配置键) ↔ Model Name (加载名称) → Model Instance

### API 端点
OpenAI Embeddings API v1 兼容：
- `GET /health` - 健康检查
- `POST /v1/embeddings` - 创建 dense embeddings
- `POST /v1/embed_all` - 创建 dense + sparse embeddings
- `GET /v1/models` - 列出模型

## CLI 命令
```bash
embed-kit serve --host 0.0.0.0 --port 8000 --model bge-m3 --config config/settings.yaml
embed-kit test-model --model bge-m3 --config config/settings.yaml
embed-kit info [--config config/settings.yaml]
embed-kit validate-config --config config/settings.yaml
```

## 开发规范
- **配置管理**：使用 Pydantic Settings，配置项必须在 YAML 中明确指定或提供默认值
- **模型开发**：实现 `BaseModelHandler` 接口并注册，使用 Pydantic BaseModel 定义输入输出
- **代码规范**：使用类型注解，遵循 PEP 8，异步处理所有 I/O 操作
