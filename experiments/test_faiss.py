
import faiss
from sentence_transformers import SentenceTransformer

# 1. 加载 MiniLM 模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. 模拟知识库
documents = [
    "Harry Potter was written by J.K. Rowling.",
    "Albert Einstein developed the theory of relativity.",
    "Isaac Newton discovered the law of universal gravitation."
]

# 3. 将知识库文本转换为向量
doc_vectors = model.encode(
    documents,
    normalize_embeddings=True
)

# 4. 创建 FAISS 索引
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)

# 5. 将文档向量加入索引
index.add(doc_vectors)

# 6. 用户提问
query = "Who wrote Harry Potter?"

# 7. 将问题转换为向量
query_vector = model.encode(
    [query],
    normalize_embeddings=True
)

# 8. 检索最相关的两条资料
scores, indices = index.search(query_vector, k=2)

for score, idx in zip(scores[0], indices[0]):
    print("参考资料：", documents[idx])
    print("相似度：", score)
    print()
