from rag_dataset import RAGDataset

dataset = RAGDataset(
    "data/RAGTruth/dataset",
    max_samples=15090
)

positive = 0
negative = 0
errors = 0

for i in range(len(dataset)):
    try:
        sample = dataset[i]
        labels = sample["labels"]

        positive += (labels == 1).sum().item()
        negative += (labels == 0).sum().item()

    except Exception as e:
        errors += 1
        print(f"第 {i} 条数据出错：{e}")

print("总样本数：", len(dataset))
print("幻觉 Token 数：", positive)
print("正常 Token 数：", negative)
print("处理失败数：", errors)