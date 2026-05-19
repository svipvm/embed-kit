import asyncio
import sys

import aiohttp


async def test_api(base_url: str = "http://localhost:8001"):
    print("=" * 70)
    print("BGE-M3 API 功能测试 (分离端点模式)")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        await test_health(session, base_url)
        await test_dense_endpoint(session, base_url)
        await test_embed_all(session, base_url)
        await test_multiple_texts(session, base_url)
        await test_models_endpoint(session, base_url)
        await test_error_handling(session, base_url)

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


async def test_health(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 1] 健康检查端点")
    try:
        async with session.get(f"{base_url}/health") as response:
            data = await response.json()
            assert response.status == 200
            assert data["status"] == "healthy"
            print("✅ 健康检查通过")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        sys.exit(1)


async def test_dense_endpoint(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 2] /v1/embeddings 端点 (只返回 dense)")
    try:
        payload = {
            "input": "This is a test sentence for dense embedding.",
            "model": "bge-m3",
            "encoding_format": "float",
        }

        async with session.post(f"{base_url}/v1/embeddings", json=payload) as response:
            data = await response.json()
            print(f"   - 响应状态: {response.status}")
            print(f"   - 响应数据: {data}")
            assert response.status == 200, f"Expected 200, got {response.status}"
            assert data["object"] == "list", f"Expected 'list', got {data.get('object')}"
            assert data["model"] == "bge-m3", f"Expected 'bge-m3', got {data.get('model')}"
            assert len(data["data"]) == 1, f"Expected 1 item, got {len(data.get('data', []))}"

            embedding = data["data"][0]
            assert embedding["object"] == "embedding", f"Expected 'embedding', got {embedding.get('object')}"
            assert embedding["index"] == 0, f"Expected 0, got {embedding.get('index')}"
            
            emb_data = embedding["embedding"]
            assert "dense" in emb_data, f"Expected 'dense' in embedding, got keys: {emb_data.keys()}"
            dense_emb = emb_data["dense"]
            assert len(dense_emb) == 1024, f"Expected 1024 dimensions, got {len(dense_emb)}"

            print(f"✅ Dense embedding 测试通过")
            print(f"   - 维度: {len(dense_emb)}")
            print(f"   - 前5个值: {dense_emb[:5]}")
    except Exception as e:
        import traceback
        print(f"❌ Dense embedding 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)


async def test_embed_all(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 3] /v1/embed_all 端点 (直接返回 dense + sparse)")
    try:
        payload = {
            "input": "This is a test for embed_all endpoint.",
            "model": "bge-m3",
            "encoding_format": "float",
        }

        async with session.post(f"{base_url}/v1/embed_all", json=payload) as response:
            data = await response.json()
            assert response.status == 200
            assert data["object"] == "list"
            assert len(data["data"]) == 1

            item = data["data"][0]
            assert item["object"] == "embedding"
            assert item["index"] == 0

            embedding = item["embedding"]
            assert "dense" in embedding
            assert "sparse" in embedding

            dense_emb = embedding["dense"]
            sparse_emb = embedding["sparse"]

            assert len(dense_emb) == 1024
            assert sparse_emb is not None
            assert "indices" in sparse_emb
            assert "values" in sparse_emb

            print(f"✅ embed_all 测试通过")
            print(f"   - Dense 维度: {len(dense_emb)}")
            print(f"   - Sparse token 数量: {len(sparse_emb['indices'])}")
            print(f"   - Sparse indices (前10个): {sparse_emb['indices'][:10]}")
            print(f"   - Sparse values (前10个): {sparse_emb['values'][:10]}")
    except Exception as e:
        print(f"❌ embed_all 测试失败: {e}")
        sys.exit(1)


async def test_multiple_texts(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 4] 多文本输入")
    try:
        payload = {
            "input": [
                "First text for embedding.",
                "Second text for embedding.",
                "Third text for embedding.",
            ],
            "model": "bge-m3",
            "encoding_format": "float",
        }

        async with session.post(f"{base_url}/v1/embed_all", json=payload) as response:
            data = await response.json()
            assert response.status == 200
            assert len(data["data"]) == 3

            for i, item in enumerate(data["data"]):
                assert item["index"] == i
                embedding = item["embedding"]
                assert len(embedding["dense"]) == 1024
                assert embedding["sparse"] is not None

            print(f"✅ 多文本输入测试通过")
            print(f"   - 文本数量: {len(data['data'])}")
            print(f"   - 每个文本 dense 维度: 1024")
            print(f"   - 每个文本都有 sparse embedding")
    except Exception as e:
        print(f"❌ 多文本输入测试失败: {e}")
        sys.exit(1)


async def test_models_endpoint(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 5] /v1/models 端点")
    try:
        async with session.get(f"{base_url}/v1/models") as response:
            data = await response.json()
            print(f"   - 响应状态: {response.status}")
            print(f"   - 响应数据: {data}")
            assert response.status == 200, f"Expected 200, got {response.status}"
            assert data["object"] == "list", f"Expected 'list', got {data.get('object')}"
            assert len(data["data"]) == 1, f"Expected 1 model, got {len(data.get('data', []))}"
            assert data["data"][0]["id"] == "bge-m3", f"Expected 'bge-m3', got {data['data'][0].get('id')}"
            assert data["data"][0]["object"] == "model", f"Expected 'model', got {data['data'][0].get('object')}"

            print(f"✅ /v1/models 端点测试通过")
            print(f"   - 模型 ID: {data['data'][0]['id']}")
            print(f"   - 模型类型: {data['data'][0]['object']}")
    except Exception as e:
        import traceback
        print(f"❌ /v1/models 端点测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)


async def test_error_handling(session: aiohttp.ClientSession, base_url: str):
    print("\n[测试 6] 错误处理")
    try:
        payload = {
            "input": "Test missing fields",
        }

        async with session.post(f"{base_url}/v1/embeddings", json=payload) as response:
            data = await response.json()
            assert response.status == 422
            assert "detail" in data

            errors = data["detail"]
            missing_fields = [err["loc"][-1] for err in errors]
            assert "model" in missing_fields
            assert "encoding_format" in missing_fields

            print(f"✅ 错误处理测试通过")
            print(f"   - 状态码: {response.status}")
            print(f"   - 缺失字段: {missing_fields}")
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_api())
