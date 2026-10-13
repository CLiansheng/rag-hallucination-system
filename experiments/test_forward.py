
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from rag_dataset import RAGDataset

# 1. 加载数据
dataset = RAGDataset("data/RAGTruth/dataset")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

batch = next(iter(dataloader))

# 2. 加载预训练 BERT + Token 分类层
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 3. 将模型和数据放到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch = {key: value.to(device) for key, value in batch.items()}

# 4. 前向传播
model.eval()

with torch.no_grad():
    outputs = model(**batch)

# 5. 查看结果
print("输入 Shape:", batch["input_ids"].shape)
print("输出 Shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())
