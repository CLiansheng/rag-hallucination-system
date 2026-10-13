
import json
from pathlib import Path
from collections import Counter

data_dir = Path("data/RAGTruth/dataset")

# 读取参考资料，建立 source_id 映射
sources = {}

with open(data_dir / "source_info.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        sources[item["source_id"]] = item

# 统计训练集
counts = Counter()
formats = {}

with open(data_dir / "response.jsonl", encoding="utf-8") as f:
    for line in f:
        response = json.loads(line)

        if response["split"] != "train":
            continue

        source = sources.get(response["source_id"])
        if source is None:
            continue

        task = source["task_type"]
        counts[task] += 1

        # 记录每种任务的参考资料格式
        if task not in formats:
            formats[task] = type(source["source_info"]).__name__

print("===== 训练集统计 =====")

for task, count in counts.items():
    print(f"{task}: {count} 条，参考资料类型: {formats[task]}")

print("训练样本总数:", sum(counts.values()))
print("\n===== 字典结构检查 =====")

seen = set()

for source in sources.values():
    task = source["task_type"]

    if task in ("QA", "Data2txt") and task not in seen:
        info = source["source_info"]

        print(f"\n{task} 字段：", info.keys())

        for key, value in info.items():
            print(f"{key}: {str(value)[:300]}")

        seen.add(task)

    if len(seen) == 2:
        break