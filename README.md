# EmbedKit

一个轻量级的 FastAPI Embedding 模型服务框架，实现 OpenAI Embeddings API v1 规范，支持多种 Embedding 模型。

## 支持的模型

### BGE-M3
- **模型ID**: `bge-m3`
- **模型名称**: BAAI/bge-m3
- **维度**: 1024
- **特性**: 支持多语言、长文本（最大8192 tokens）、Dense + Sparse Embeddings
- **适用场景**: 文档检索、语义搜索、文本相似度计算

## 特性

- 🚀 FastAPI 高性能异步框架
- 🔌 OpenAI Embeddings API v1 兼容
- 📦 使用 uv 进行包管理
- 🐳 支持 Docker 容器化部署
- 💻 CLI 一键启动
- ⚙️ YAML 配置文件管理
- 🎯 支持 Dense 和 Sparse Embeddings
- 🔒 Pydantic Settings 配置验证

## 安装

### 使用 uv（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd embed-kit

# 安装 PyTorch（CUDA 12.8）
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 安装项目依赖（包含 pydantic-settings）
uv sync
```

### 使用 pip

```bash
pip install -e .
```

## 快速启动

### 1. CLI 启动服务

```bash
# 基本启动（使用 bge-m3 模型）
uv run embed-kit serve --host 0.0.0.0 --port 8000 --model bge-m3 --config config/settings.yaml

# 指定端口
uv run embed-kit serve --host 0.0.0.0 --port 8080 --model bge-m3 --config config/settings.yaml

# 开发模式（自动重载）
uv run embed-kit serve --host 0.0.0.0 --port 8000 --model bge-m3 --config config/settings.yaml --reload

# 指定日志级别
uv run embed-kit serve --host 0.0.0.0 --port 8000 --model bge-m3 --config config/settings.yaml --log-level DEBUG
```

### 2. Docker 启动

```bash
# 生产环境
docker-compose up embed-kit

# 开发环境（支持热重载）
docker-compose up embed-kit-dev
```

### 3. Python 直接启动

```bash
source .venv/bin/activate
python -m embed_kit.cli serve --host 0.0.0.0 --port 8000 --model bge-m3 --config config/settings.yaml
```

## CLI 命令

### serve - 启动服务

```bash
uv run embed-kit serve --host <host> --port <port> --model <model-name> --config <config-file> [OPTIONS]

必需参数:
  --host TEXT              绑定主机
  --port INTEGER           绑定端口
  --model TEXT             模型名称
  --config TEXT            配置文件路径

可选参数:
  --reload                 启用自动重载（开发模式）
  --log-level TEXT         日志级别 [默认: INFO]
```

### test-model - 测试模型

```bash
uv run embed-kit test-model --model bge-m3 --config config/settings.yaml
```

### info - 显示系统信息

```bash
# 显示已注册的处理器
uv run embed-kit info

# 显示处理器和配置的模型
uv run embed-kit info --config config/settings.yaml
```

### validate-config - 验证配置文件

```bash
uv run embed-kit validate-config --config config/settings.yaml
```

## API 调用

启动服务后访问 API 文档: http://localhost:8000/docs

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

响应:
```json
{
  "status": "healthy",
  "model": "bge-m3",
  "app": "EmbedKit",
  "version": "0.1.0"
}
```

### 2. 创建 Dense Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test sentence.",
    "model": "bge-m3",
    "encoding_format": "float"
  }'
```

响应:
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": {
      "dense": [0.0023, -0.0093, ...],
      "sparse": {"indices": [], "values": []}
    },
    "index": 0
  }],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### 3. 创建 Dense + Sparse Embeddings

```bash
curl -X POST http://localhost:8000/v1/embed_all \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test sentence.",
    "model": "bge-m3",
    "encoding_format": "float"
  }'
```

响应:
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": {
      "dense": [0.0023, -0.0093, ...],
      "sparse": {
        "indices": [3293, 83, 10, ...],
        "values": [0.1838, 0.1272, 0.0679, ...]
      }
    },
    "index": 0
  }],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### 4. 批量处理

```bash
curl -X POST http://localhost:8000/v1/embed_all \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First text.", "Second text.", "Third text."],
    "model": "bge-m3",
    "encoding_format": "float"
  }'
```

### 5. 列出模型

```bash
curl http://localhost:8000/v1/models
```

响应:
```json
{
  "object": "list",
  "data": [{
    "id": "bge-m3",
    "object": "model",
    "owned_by": "embed-kit"
  }]
}
```

### 6. Python 调用示例

```python
import aiohttp
import asyncio

async def get_embeddings():
    async with aiohttp.ClientSession() as session:
        # Dense embeddings
        async with session.post(
            "http://localhost:8000/v1/embeddings",
            json={
                "input": "Hello, world!",
                "model": "bge-m3",
                "encoding_format": "float"
            }
        ) as response:
            data = await response.json()
            print(data)

asyncio.run(get_embeddings())
```

## 配置

使用 **pydantic-settings** 进行配置管理，提供类型安全和自动验证功能。

### 配置文件 (config/settings.yaml)

```yaml
app:
  name: EmbedKit              # 应用名称
  version: 0.1.0              # 应用版本
  default_log_level: INFO     # 默认日志级别

models:
  bge-m3:                          # Model ID（后端定位模型）
    name: BAAI/bge-m3              # Model Name（前端调用和实际加载）
    path: checkpoints/BAAI/bge-m3  # 模型文件路径
    use_fp16: true                  # 使用 FP16 加速
    device: cuda                    # 设备类型 (cuda/cpu)
    normalize_embeddings: true      # 归一化 embeddings
    batch_size: 12                  # 批处理大小
    max_length: 8192                # 最大序列长度
```

### 配置管理特性

- **类型安全**: 使用 Pydantic 进行严格的类型检查和验证
- **自动验证**: 字段验证、必填检查、类型转换自动完成
- **易于扩展**: 添加新配置项只需在配置类中添加字段
- **IDE 支持**: Pydantic 模型有良好的 IDE 自动补全支持

### 配置说明

#### 应用配置 (`app`)
- `name`: 应用名称，显示在健康检查和日志中
- `version`: 应用版本，用于版本管理和 API 响应
- `default_log_level`: 默认日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

#### 模型配置 (`models`)
- **Model ID** (`bge-m3`): 配置键名，用于 CLI 参数和 API 调用
- **Model Name** (`name`): 实际模型名称，用于加载模型（如 `BAAI/bge-m3`）
- **Model Path** (`path`): 模型文件路径（必需）
- `batch_size`: 批处理大小，根据 GPU 内存调整（必需，>= 1）
- `max_length`: 最大序列长度（必需，>= 1）
- `use_fp16`: 使用 FP16 加速（默认 true）
- `device`: 设备类型 cuda/cpu（默认 cuda）
- `normalize_embeddings`: 归一化 embeddings（默认 true）

如果省略 `name`，会自动从 `path` 推断（如 `checkpoints/BAAI/bge-m3` → `BAAI/bge-m3`）

### 配置使用示例

```python
from embed_kit.utils.config import Settings

# 加载配置
settings = Settings.from_yaml("config/settings.yaml")

# 访问应用配置
print(f"App: {settings.app.name} v{settings.app.version}")
print(f"Log Level: {settings.app.default_log_level}")

# 访问模型配置
model_config = settings.get_model_config("bge-m3")
print(f"Model: {model_config.get_model_name()}")
print(f"Path: {model_config.path}")
print(f"Batch Size: {model_config.batch_size}")
print(f"Max Length: {model_config.max_length}")
```

**多模型配置示例:**

```yaml
app:
  name: EmbedKit
  version: 0.1.0
  default_log_level: INFO

models:
  bge-m3:
    name: BAAI/bge-m3
    path: checkpoints/BAAI/bge-m3
    use_fp16: true
    device: cuda
    normalize_embeddings: true
    batch_size: 12
    max_length: 8192
  
  bge-large-zh:
    name: BAAI/bge-large-zh-v1.5
    path: checkpoints/BAAI/bge-large-zh-v1.5
    use_fp16: true
    device: cuda
    normalize_embeddings: true
    batch_size: 32
    max_length: 512
  
  e5-large:
    path: checkpoints/intfloat/e5-large-v2  # name 会自动推断为 intfloat/e5-large-v2
    use_fp16: true
    device: cuda
    normalize_embeddings: true
    batch_size: 32
    max_length: 512
```

**映射关系:**
```
Model ID (CLI/API)  ↔  Model Name (加载)  →  Model Instance
     bge-m3         ↔     BAAI/bge-m3     →  BGEM3Handler
```

## 测试

### 运行测试脚本

```bash
# 启动服务
uv run embed-kit serve --host 0.0.0.0 --port 8001 --model bge-m3 --config config/settings.yaml

# 运行测试（另一个终端）
source .venv/bin/activate
python test/test_api.py
```

### 测试覆盖内容
- ✅ 健康检查端点
- ✅ Dense embedding 生成
- ✅ Dense + Sparse embedding 生成
- ✅ 多文本批量处理
- ✅ 模型列表查询
- ✅ 错误处理验证

## 项目结构

```
embed-kit/
├── src/embed_kit/
│   ├── main.py              # FastAPI 应用入口
│   ├── cli.py               # CLI 命令定义
│   ├── models/              # 模型处理器
│   │   ├── base.py          # 基础接口
│   │   ├── embedding.py     # Embedding 实现
│   │   └── registry.py      # 模型注册表
│   ├── api/                 # API 路由
│   │   ├── routes.py        # OpenAI 兼容路由
│   │   └── schemas.py       # 请求/响应模型
│   └── utils/               # 工具函数
│       ├── config.py        # 配置管理（pydantic-settings）
│       └── logger.py        # 日志配置
├── config/
│   └── settings.yaml        # 应用配置
├── test/
│   └── test_api.py          # API 测试脚本
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── AGENT.md                 # 开发约束规范
└── README.md
```

## 扩展开发

### 添加新模型

详见 [AGENT.md](AGENT.md) 了解如何添加自定义模型处理器。

### 修改配置

配置文件使用 Pydantic Settings 管理，修改配置时：

1. **修改配置文件** (`config/settings.yaml`)
   - 添加新的模型配置
   - 修改应用参数

2. **扩展配置类** (如需要新字段)
   - 在 `src/embed_kit/utils/config.py` 中添加字段
   - Pydantic 会自动验证新字段

3. **使用配置**
   ```python
   from embed_kit.utils.config import Settings
   
   settings = Settings.from_yaml("config/settings.yaml")
   # 访问配置项
   ```

## 许可证

MIT
