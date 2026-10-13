import json
from pathlib import Path

# 参考资料文件
path = Path("data/RAGTruth/dataset/source_info.jsonl")

# 读取第一条数据
with open(path, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

# 查看字段和完整内容
print("字段：", sample.keys())
print("\n完整数据：")
print(json.dumps(sample, indent=2, ensure_ascii=False))