import json
from pathlib import Path

path = Path("data/RAGTruth/dataset/response.jsonl")

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        # 找到包含幻觉标注的样本
        if sample["labels"]:
            print("LLM 回答：")
            print(sample["response"])

            print("\n幻觉标注：")
            print(json.dumps(sample["labels"], indent=2, ensure_ascii=False))
            break