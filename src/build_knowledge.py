import json
from pathlib import Path

data_dir = Path("data/RAGTruth/dataset")
documents = []
article_count = 0
with open(data_dir / "source_info.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        # 第一版只使用 Summary 文章
        if item["task_type"] != "Summary":
            continue

        text = item["source_info"]

        # 将长文章切成较短的文本块
        for start in range(0, len(text), 400):
            chunk = text[start:start + 400]

            if chunk.strip():
                documents.append(chunk)

        # 暂时只使用 20 篇文章
        article_count += 1
        if article_count >= 20:
            break

# 保存知识库文本
output = Path("data/knowledge.json")
output.write_text(
    json.dumps(documents, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("文章数量：", article_count)
print("知识库文本块数量：", len(documents))