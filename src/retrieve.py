
import os
import json
import torch
import faiss

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer


# ==================== 1. 初始化 ====================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载知识库与 FAISS
documents = json.loads(
    (ROOT / "data/knowledge.json").read_text(encoding="utf-8")
)
index = faiss.read_index(str(ROOT / "data/knowledge.index"))

# 加载 MiniLM
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 加载微调后的 BERT
model_path = ROOT / "models/bert_hallucination"

tokenizer = AutoTokenizer.from_pretrained(model_path)
bert = AutoModelForTokenClassification.from_pretrained(model_path)
bert = bert.to(device)
bert.eval()

# 创建 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


# ==================== 2. RAG 检索 ====================

def retrieve_documents(query, k=3):
    # 用户问题转换成向量
    query_vector = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # FAISS 检索
    scores, indices = index.search(
        query_vector,
        k=min(k, index.ntotal)
    )

    # 返回检索到的原始文本
    results = [documents[int(idx)] for idx in indices[0]]

    return "\n\n".join(results)


# ==================== 3. LLM 生成回答 ====================

def generate_answer(query, context):
    prompt = f"""
Answer the question in English.
Use only the reference materials.
If evidence is insufficient, express uncertainty.

Reference materials:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ==================== 4. BERT 幻觉定位 ====================

def detect_hallucination(context, answer):
    encoding = tokenizer(
        context,
        answer,
        truncation="longest_first",
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids(0)

    # offset_mapping 不能直接输入 BERT
    encoding.pop("offset_mapping")

    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = bert(**inputs)

    preds = outputs.logits.argmax(dim=-1)[0].tolist()

    # 合并相邻的可疑 Token
    spans = []

    for pred, seq_id, (start, end) in zip(
        preds, sequence_ids, offsets
    ):
        if seq_id != 1 or pred != 1 or start == end:
            continue

        if spans and start <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    return [answer[start:end] for start, end in spans]


# ==================== 5. LLM 核验与纠错 ====================

def correct_answer(query, context, answer):
    prompt = f"""
You are a factual verification assistant.

Reference materials:
{context}

Question:
{query}

Original answer:
{answer}

Instructions:
1. Compare the answer with the reference materials.
2. Identify contradictions or unsupported claims.
3. Correct errors using only the reference materials.
4. Express uncertainty when evidence is insufficient.
5. Keep the answer unchanged if no correction is needed.
6. Respond in English.

Output the corrected answer only.
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ==================== 6. 主程序 ====================

def main():
    query = input("请输入问题：").strip()

    if not query:
        print("问题不能为空")
        return

    # 1. 检索
    context = retrieve_documents(query)

    print("\n===== 检索资料 =====")
    print(context)

    # 2. 生成
    answer = generate_answer(query, context)

    print("\n===== LLM 原始回答 =====")
    print(answer)

    # 3. 幻觉定位
    hallucinations = detect_hallucination(context, answer)

    print("\n===== BERT 可疑幻觉片段 =====")
    print(hallucinations)

    # 4. 核验与纠错
    corrected = correct_answer(query, context, answer)

    print("\n===== 纠错后回答 =====")
    print(corrected)


if __name__ == "__main__":
    main()
