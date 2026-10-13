
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from rag_dataset import RAGDataset

# 1. 准备训练数据
dataset = RAGDataset("data/RAGTruth/dataset", max_samples=15090)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 2. 加载预训练 BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
).to(device)

# 3. 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 4. 开始训练
model.train()
total_loss = 0
steps = 0

for i, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}

    # 跳过没有有效监督 Token 的样本
    if not (batch["labels"] != -100).any().item():
        continue

    optimizer.zero_grad()

    # 前向传播
    outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch["token_type_ids"]
    )

    # 提高幻觉 Token 的损失权重
    weights = torch.tensor([1.0, 20.0], device=device)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100
    )

    loss = loss_fn(
        outputs.logits.reshape(-1, 2),
        batch["labels"].reshape(-1)
    )

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    total_loss += loss.item()
    steps += 1

    if (i + 1) % 20 == 0:
        print(f"Step {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

print(f"训练完成！平均 Loss: {total_loss / steps:.4f}")

# 5. 保存微调后的模型和分词器
save_path = Path("models/bert_hallucination")
save_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(save_path)
dataset.tokenizer.save_pretrained(save_path)

print("模型已保存到：", save_path)
