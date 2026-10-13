from rag_dataset import RAGDataset

dataset = RAGDataset("data/RAGTruth/dataset", max_samples=200)

positive = 0
negative = 0

for sample in dataset:
    labels = sample["labels"]

    positive += (labels == 1).sum().item()
    negative += (labels == 0).sum().item()

print("幻觉 Token 数：", positive)
print("正常 Token 数：", negative)
print("幻觉占比：", positive / (positive + negative))