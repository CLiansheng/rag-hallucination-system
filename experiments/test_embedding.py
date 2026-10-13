from sentence_transformers import SentenceTransformer

# 加载预训练 MiniLM 模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 准备两段文本
sentences = [
    "Who wrote Harry Potter?",
    "Harry Potter was written by J.K. Rowling."
]

# 将文本转换为向量
embeddings = model.encode(sentences)

print("向量 Shape:", embeddings.shape)
print("第一个句子的前10维:", embeddings[0][:10])
from sklearn.metrics.pairwise import cosine_similarity

# 计算两个句子向量的余弦相似度
similarity = cosine_similarity(embeddings)

print("相似度矩阵：")
print(similarity)