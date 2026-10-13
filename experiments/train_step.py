import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from rag_dataset import RAGDataset

# 1. 加载数据
dataset = RAGDataset("data/RAGTruth/dataset")
loader = DataLoader(dataset, batch_size=1, shuffle=True)
batch = next(iter(loader))

# 2. 加载 BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
).to(device)

batch = {k: v.to(device) for k, v in batch.items()}

# 3. 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 4. 训练一次
model.train()
optimizer.zero_grad()

outputs = model(**batch)
loss = outputs.loss

loss.backward()
optimizer.step()

print("训练完成！")
print("Loss:", loss.item())
