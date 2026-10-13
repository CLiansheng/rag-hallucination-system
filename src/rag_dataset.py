import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def format_source(source):
    task = source["task_type"]
    info = source["source_info"]

    if task == "Summary":
        return info

    elif task == "QA":
        return (
            f"Question: {info['question']}\n"
            f"Passages: {info['passages']}"
        )

    elif task == "Data2txt":
        return json.dumps(info, ensure_ascii=False)

    else:
        raise ValueError(f"未知任务类型: {task}")
class RAGDataset(Dataset):
    def __init__(self, data_dir, max_samples=200,split="train"):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_dir = Path(data_dir)

        # 建立 source_id 到参考资料的映射
        sources = {}
        with open(data_dir / "source_info.jsonl", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                sources[item["source_id"]] = item

        # 收集训练样本
        self.samples = []
        with open(data_dir / "response.jsonl", encoding="utf-8") as f:
            for line in f:
                response = json.loads(line)
                source = sources.get(response["source_id"])

                # 第一版只使用摘要任务
                if response["split"] != split or source is None:
                    continue


                self.samples.append((format_source(source), response))

                if len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        evidence, response = self.samples[idx]

        # 编码参考资料和回答
        encoding = self.tokenizer(
            evidence,
            response["response"],
            truncation="longest_first",
            max_length=512,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding["offset_mapping"][0].tolist()
        sequence_ids = encoding.sequence_ids(0)

        # 生成 Token 级幻觉标签
        labels = []
        for seq_id, (start, end) in zip(sequence_ids, offsets):
            if seq_id != 1:
                labels.append(-100)
                continue

            label = 0
            for h in response["labels"]:
                if h.get("implicit_true", False):
                    continue

                if start < h["end"] and end > h["start"]:
                    label = 1
                    break

            labels.append(label)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "token_type_ids": encoding["token_type_ids"][0],
            "labels": torch.tensor(labels, dtype=torch.long)
        }


if __name__ == "__main__":
    dataset = RAGDataset("data/RAGTruth/dataset",max_samples=15090)

    print("样本数量：", len(dataset))

    sample = dataset[0]

    for key, value in sample.items():
        print(key, value.shape)