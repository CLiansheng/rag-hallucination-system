
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. 读取知识库
path = Path("data/knowledge.json")
documents = json.loads(path.read_text(encoding="utf-8"))

# 2. 加载 MiniLM
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. 将所有 Chunk 转换成向量
vectors = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

# 4. 创建 FAISS 索引
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)

# 5. 将向量加入索引
index.add(vectors)

# 6. 保存索引到本地
faiss.write_index(index, "data/knowledge.index")

print("向量 Shape:", vectors.shape)
print("索引中的向量数:", index.ntotal)
