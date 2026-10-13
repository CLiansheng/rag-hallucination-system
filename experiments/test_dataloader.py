from torch.utils.data import DataLoader
from rag_dataset import RAGDataset

# 创建数据集
dataset = RAGDataset("data/RAGTruth/dataset")

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# 取出第一个 Batch
batch = next(iter(dataloader))

# 查看张量形状
for key, value in batch.items():
    print(key, value.shape)