
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from rag_dataset import RAGDataset

# 1. 加载测试集（未参与训练）
dataset = RAGDataset(
    "data/RAGTruth/dataset",
    max_samples=2700,
    split="test"
)
loader = DataLoader(dataset, batch_size=1)

# 2. 加载微调后的 BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained(
    "models/bert_hallucination"
).to(device)

model.eval()

# 3. 统计预测结果
TP, FP, FN = 0, 0, 0

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        # [B, 512, 2] -> [B, 512]
        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # 忽略 -100 标签
        valid = labels != -100

        TP += ((preds == 1) & (labels == 1) & valid).sum().item()
        FP += ((preds == 1) & (labels == 0) & valid).sum().item()
        FN += ((preds == 0) & (labels == 1) & valid).sum().item()

# 4. 计算指标
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("测试样本数：", len(dataset))
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print("真实幻觉 Token 数：", TP + FN)
print("预测幻觉 Token 数：", TP + FP)