import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 加载预训练 BERT，并添加 Token 二分类层
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("运行设备：", device)
print("模型类型：", model.__class__.__name__)
print("分类类别数：", model.config.num_labels)