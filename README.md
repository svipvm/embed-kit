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

## 安装

### 使用 uv（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd embed-kit

# 安装 PyTorch（CUDA 12.8）
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 安装项目依赖
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
uv run embed-kit serve --model bge-m3 --config config/models.yaml

# 指定端口
uv run embed-kit serve --model bge-m3 --config config/models.yaml --port 8080

# 开发模式（自动重载）
uv run embed-kit serve --model bge-m3 --config config/models.yaml --reload

# 多进程模式
uv run embed-kit serve --model bge-m3 --config config/models.yaml --workers 4

# 指定日志级别
uv run embed-kit serve --model bge-m3 --config config/models.yaml --log-level DEBUG
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
python -m embed_kit.cli serve --model bge-m3 --config config/models.yaml
```

## CLI 命令

### serve - 启动服务

```bash
uv run embed-kit serve --model <model-name> --config <config-file> [OPTIONS]

选项:
  --host TEXT              绑定主机 [默认: 0.0.0.0]
  --port INTEGER           绑定端口 [默认: 8000]
  --workers INTEGER        工作进程数 [默认: 1]
  --reload                 启用自动重载（开发模式）
  --log-level TEXT         日志级别 [默认: INFO]
  --model TEXT             模型名称 [必需]
  --config TEXT            配置文件路径 [必需]
```

### test-model - 测试模型

```bash
uv run embed-kit test-model --model bge-m3 --config config/models.yaml
```

### list-models - 列出可用模型

```bash
uv run embed-kit list-models --config config/models.yaml
```

### list-handlers - 列出已注册处理器

```bash
uv run embed-kit list-handlers
```

### validate-config - 验证配置文件

```bash
uv run embed-kit validate-config --config config/models.yaml
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

### 模型配置文件 (config/models.yaml)

```yaml
models:
  bge-m3:
    type: bge-m3
    config:
      model_name: BAAI/bge-m3        # 模型名称
      model_path: checkpoints/BAAI/bge-m3  # 模型路径
      use_fp16: true                  # 使用 FP16 加速
      device: cuda                    # 设备类型 (cuda/cpu)
      normalize_embeddings: true      # 归一化 embeddings
      batch_size: 12                  # 批处理大小
      max_length: 8192                # 最大序列长度
```

**重要配置项说明:**
- `model_name`: 实际模型名称（如 BAAI/bge-m3）
- `model_path`: 模型文件路径
- `batch_size`: 批处理大小，根据 GPU 内存调整
- `max_length`: 最大序列长度，BGE-M3 支持 8192

## 测试

### 运行测试脚本

```bash
# 启动服务
uv run embed-kit serve --model bge-m3 --config config/models.yaml --port 8001

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
│       └── logger.py        # 日志配置
├── config/
│   └── models.yaml          # 模型配置
├── test/
│   └── test_api.py          # API 测试脚本
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── AGENT.md                 # 开发约束规范
└── README.md
```

## 环境变量

服务启动时会设置以下环境变量：

- `EMBED_KIT_APP_NAME` - 应用名称
- `EMBED_KIT_APP_VERSION` - 应用版本
- `EMBED_KIT_SELECTED_MODEL` - 选择的模型
- `EMBED_KIT_MODELS_CONFIG_PATH` - 配置文件路径
- `EMBED_KIT_HOST` - 服务主机
- `EMBED_KIT_PORT` - 服务端口
- `EMBED_KIT_WORKERS` - 工作进程数

## 扩展开发

详见 [AGENT.md](AGENT.md) 了解如何添加自定义模型处理器。

## 许可证

MIT
