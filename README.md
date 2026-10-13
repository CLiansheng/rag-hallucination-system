# RAG + LLM 幻觉检测、定位与纠错系统

这是一个基于 **RAG（检索增强生成）**、**BERT Token 分类**与 **大语言模型（LLM）** 的 NLP 实验项目。系统先从知识库检索参考证据，再调用 LLM 生成回答，随后用微调后的 BERT 标记可疑幻觉片段，并利用 LLM 对照证据进行核验和修正。

> **项目状态：** 第一版端到端流程已跑通。BERT 幻觉定位模型仅作为实验性基线，**目前不能作为可靠的事实核验工具**。

## 一、系统架构

```text
用户输入问题
    │
    ▼
MiniLM 文本向量化
    │
    ▼
FAISS 相似度检索 ──────► 参考资料
                              │
                              ▼
                       DeepSeek 生成回答
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            微调后的 BERT          DeepSeek 证据核验
            Token 级幻觉定位          与自动纠错
                    │                   │
                    ▼                   ▼
              可疑幻觉片段           修正后的回答
```

目前，**BERT 的可疑片段输出与 LLM 的核验纠错并行运行**；纠错模块会独立查看参考资料和原始回答，并非仅在 BERT 检测到幻觉时才启动。

## 二、核心模块

| 文件 | 作用 |
| --- | --- |
| `src/rag_dataset.py` | 读取 RAGTruth，匹配参考资料与 LLM 回答，将字符级幻觉标注映射为 Token 标签 |
| `src/train.py` | 微调 `bert-base-uncased`，训练 Token 二分类模型 |
| `src/test.py` | 计算幻觉 Token 的 Precision、Recall 和 F1 |
| `src/build_knowledge.py` | 提取参考文章并切分文本块（Chunk） |
| `src/build_index.py` | 使用 `all-MiniLM-L6-v2` 生成向量并建立 FAISS 索引 |
| `src/retrieve.py` | 运行完整流程：检索、生成、BERT 定位、LLM 纠错 |

BERT 的输入形式为：

```text
[CLS] 参考证据 [SEP] LLM 回答 [SEP]
```

每个 Token 的训练标签含义如下：

- `0`：回答中未被标注为幻觉的 Token。
- `1`：回答中被人工标注为幻觉的 Token。
- `-100`：参考证据、特殊 Token 或 Padding，不参与分类损失计算。

## 三、实验结果

使用 RAGTruth 数据集微调 BERT，并在独立测试集上评估。

| 配置或指标 | 结果 |
| --- | --- |
| 编码模型 | `bert-base-uncased` |
| 训练样本数 | 15,090 |
| 测试样本数 | 2,700 |
| 训练轮数 | 1 Epoch |
| Batch Size | 1 |
| 学习率 | `2e-5` |
| 损失函数 | 加权交叉熵，类别权重 `[1.0, 20.0]` |
| 最大输入长度 | 512 Token |
| **Token 级 Precision** | **0.2085（20.85%）** |
| **Token 级 Recall** | **0.5275（52.75%）** |
| **Token 级 F1** | **0.2989（29.89%）** |

以上是**一次本地实验的结果**，不代表论文基准成绩。评估时排除了标签为 `-100` 的位置；模型仍存在较多误报和漏报，因此 BERT 输出只能视为**可疑片段**，不能直接认定为事实错误。

## 四、项目目录

```text
rag_hallucination_system/
├── src/
│   ├── rag_dataset.py       # RAGTruth 数据集处理
│   ├── train.py             # BERT 微调
│   ├── test.py              # Token 级评估
│   ├── build_knowledge.py   # 文章分块
│   ├── build_index.py       # MiniLM + FAISS 索引
│   └── retrieve.py          # 完整系统入口
├── experiments/             # 学习和调试阶段的测试脚本（可选）
├── data/                    # 数据集、知识库和索引（不上传）
├── models/                  # 微调模型权重（不上传）
├── .env.example             # API Key 配置模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 五、运行与复现

以下命令以 **Windows PowerShell + Python 3.12** 为例，均在项目根目录运行。

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv rag_nlp
.\rag_nlp\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需使用 NVIDIA GPU，应安装与系统及驱动兼容的 PyTorch CUDA 版本。本项目开发环境使用的是 **PyTorch `2.10.0+cu128`**。可参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)；仅用 CPU 也可以运行，但训练速度会较慢。

### 2. 下载 RAGTruth 数据集

```powershell
New-Item -ItemType Directory -Force data | Out-Null
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth
```

确保以下文件存在：

```text
data/RAGTruth/dataset/response.jsonl
data/RAGTruth/dataset/source_info.jsonl
```

### 3. 构建 RAG 知识库

```powershell
python src/build_knowledge.py
python src/build_index.py
```

当前 Demo 从 `Summary` 任务中选取前 **20 篇参考文章**，按 **400 个字符**切分为不重叠的文本块。在本地实验中，共生成 **181 个 Chunk**。运行后生成：

```text
data/knowledge.json
data/knowledge.index
```

### 4. 微调并测试 BERT

```powershell
python src/train.py
python src/test.py
```

训练完成后，模型默认保存至：

```text
models/bert_hallucination/
```

`rag_dataset.py` 支持 `Summary`、`QA` 和 `Data2txt` 三类任务。为复现上述全量实验，请确认 `train.py` 的 `max_samples=15090`，`test.py` 的 `max_samples=2700`，并使用相同的训练参数。

> 由于 GitHub 仓库不包含微调后的模型权重，首次运行完整 Demo 前，需要先执行训练脚本，或自行将兼容的模型权重放到指定目录。

### 5. 配置 DeepSeek API

复制配置模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入你自己的密钥：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**不要公开或提交真实的 API Key。** 调用 DeepSeek API 可能产生费用。

### 6. 运行完整系统

```powershell
python src/retrieve.py
```

示例问题：

```text
When did Anne Frank die?
```

程序将依次输出：**检索到的参考资料、LLM 原始回答、BERT 标出的可疑片段、证据核验后的回答**。

## 六、当前局限性

1. **幻觉定位精度有限**：Token 级 F1 为 0.2989，BERT 的输出仅供辅助参考。
2. **上下文长度受限**：BERT 最大处理 512 个 Token；截断可能丢失关键证据或部分回答。
3. **知识库较小且分块简单**：仅使用 20 篇文章，按字符固定切块，可能截断单词和句子。
4. **纠错不保证正确**：生成与核验均使用 DeepSeek，存在重复错误或引入新错误的可能；目前缺少独立的纠错效果评估。
5. **指标是 Token 级而非片段级**：尚未评估严格的幻觉片段边界准确率。

## 七、数据与安全说明

本仓库计划只上传**源代码、依赖文件和配置模板**。通过 `.gitignore` 排除原始数据、FAISS 索引、模型权重、虚拟环境以及包含密钥的 `.env`。

数据集来源：[RAGTruth 官方仓库](https://github.com/ParticleMedia/RAGTruth)。如需重新分发数据，请遵守其原始许可证和使用条款。

## 八、后续改进方向

后续可以研究：更合理的证据选择与长文本处理、类别不平衡优化、不同任务之间的泛化能力、Token/Span 级精细定位，以及独立的事实核验机制。

这些属于**后续研究方向**，并非当前版本已经实现的功能。
