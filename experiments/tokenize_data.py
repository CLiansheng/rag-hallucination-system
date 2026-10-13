import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
path = Path("data/RAGTruth/dataset/response.jsonl")

# 读取一条存在幻觉的样本
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        if sample["labels"]:
            break

# 对完整回答分词
encoding = tokenizer(
    sample["response"],
    add_special_tokens=False,
    return_offsets_mapping=True,
    truncation=False
)

tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
offsets = encoding["offset_mapping"]

# 为每个 Token 生成标签
labels = [0] * len(tokens)

for hallucination in sample["labels"]:
    if hallucination.get("implicit_true", False):
        continue

    h_start = hallucination["start"]
    h_end = hallucination["end"]

    for i, (start, end) in enumerate(offsets):
        if start < h_end and end > h_start:
            labels[i] = 1

# 只输出被标记为幻觉的 Token
for token, label in zip(tokens, labels):
    if label == 1:
        print(token, label)

print("Token 总数：", len(tokens))
print("幻觉 Token 数：", sum(labels))