import json
from pathlib import Path

data_dir = Path("data/RAGTruth/dataset")

# 1. 读取参考资料，建立字典
sources = {}

with open(data_dir / "source_info.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        sources[item["source_id"]] = item

# 2. 找到一条带有幻觉的回答
with open(data_dir / "response.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        response = json.loads(line)

        if response["split"] == "train" and response["labels"]:
            break

# 3. 根据 source_id 找到对应的参考资料
source = sources[response["source_id"]]

print("source_id:", response["source_id"])
print("任务类型:", source["task_type"])
print("参考资料:", str(source["source_info"])[:500])
print("\nLLM 回答:", response["response"][:500])
print("\n幻觉数量:", len(response["labels"]))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 获取参考资料和 LLM 回答
evidence = source["source_info"]
answer = response["response"]

# 将两段文本一起编码
encoding = tokenizer(
    evidence,
    answer,
    truncation="only_first",
    max_length=512,
    return_offsets_mapping=True,
    return_tensors="pt"
)

# 每个 Token 对应原文的字符位置
offsets = encoding["offset_mapping"][0].tolist()

# 区分参考资料和回答
sequence_ids = encoding.sequence_ids(0)

labels = []

for seq_id, (start, end) in zip(sequence_ids, offsets):

    # 参考资料和特殊 Token 不参与训练
    if seq_id != 1:
        labels.append(-100)
        continue

    # 默认认为回答 Token 正常
    label = 0

    # 与人工幻觉标注比较
    for h in response["labels"]:
        if h.get("implicit_true", False):
            continue

        if start < h["end"] and end > h["start"]:
            label = 1
            break

    labels.append(label)

print("输入 Token 数：", len(labels))
print("幻觉 Token 数：", labels.count(1))
print("正常回答 Token 数：", labels.count(0))
print("忽略 Token 数：", labels.count(-100))

import torch

# 将标签列表转换为 PyTorch Tensor
labels = torch.tensor([labels], dtype=torch.long)

print("input_ids:", encoding["input_ids"].shape)
print("labels:", labels.shape)