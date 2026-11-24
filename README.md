# TransE

使用 PyTorch 实现的 TransE 知识图谱嵌入模型，支持在 FB15k 数据集上训练，并使用 Raw / Filtered Mean Rank 和 Hits@10 评估链接预测效果。

## 功能

- 从 FB15k 文本文件读取实体、关系和三元组
- 训练时随机替换头实体或尾实体进行负采样，并避免生成训练集中的正例
- 支持 L1、L2 距离以及 CPU、CUDA 训练
- 保存模型参数、优化器状态、训练配置和指标
- 定期在验证集上评估，按 Raw MR 保存最佳模型 `best.pt`，同时保留最近模型 `last.pt`
- 支持头实体和尾实体预测，以及 Raw / Filtered MR 和 Hits@10

## 项目结构

```text
TransE/
├── .gitignore
├── model.py          # TransE 模型
├── dataset.py        # FB15k 数据读取、负采样和批次生成
├── train.py          # 训练、验证与检查点保存
├── evaluate.py       # 链接预测评估
├── requirements.txt  # Python 依赖
├── README.md
└── data/
    └── FB15k/        # FB15K数据集
        ├── entity2id.txt
        ├── relation2id.txt
        ├── train.txt
        ├── valid.txt
        └── test.txt
```

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.0 或更高版本

创建并激活虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash：

```bash
source .venv/Scripts/activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要特定的 CUDA 版本，请按照 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 提供的命令安装对应版本。虚拟环境和 PyTorch 的本地安装文件不应提交到 Git。

## 准备数据

数据目录需要包含以下五个文件：

```text
FB15k/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── valid.txt
└── test.txt
```

其中实体和关系映射文件使用制表符分隔名称与 ID；三元组文件依次包含头实体、尾实体和关系，同样使用制表符分隔。

数据集不包含在仓库中。请将 `/path/to/FB15k` 替换为实际数据目录，或把数据放在项目内的 `data/FB15k/` 后使用 `--data-dir data/FB15k`。`data/` 已被 `.gitignore` 忽略。

可以先检查数据能否正常加载：

```bash
python dataset.py --data-dir /path/to/FB15k
```

## 训练

```bash
python train.py --data-dir /path/to/FB15k --output-dir runs/TransE
```

默认训练 1000 轮，嵌入维度为 50，批次大小为 1024，使用 L1 距离、1.0 的间隔和 0.01 的学习率。反向传播使用批次求和损失，与论文算法一致；日志中的 `loss` 仍是每条三元组的平均损失，便于观察。

每 10 轮以及最后一轮会在验证集前 1000 条三元组上评估，按验证集 Raw MR 选择最佳模型。检查点保存为：

- `runs/TransE/last.pt`：最近一轮的模型。
- `runs/TransE/best.pt`：验证集 Raw MR 最低的模型。

常用参数：

```bash
python train.py --data-dir /path/to/FB15k --output-dir runs/TransE --epochs 1000 --batch-size 1024 --embedding-dim 50 --distance l1 --learning-rate 0.01 --margin 1.0 --validation-interval 10 --validation-limit 1000
```

程序会根据当前解释器的 PyTorch 环境自动选择设备：CUDA 可用时使用 GPU，否则使用 CPU，不需要额外参数。将 `--validation-limit` 设为 `0` 可以使用完整验证集，但每次评估会更慢。

重复使用同一个 `--output-dir` 会覆盖其中的 `last.pt` 和 `best.pt`，需要保留历史模型时应先复制检查点。运行产物所在的 `runs/` 不会被提交。

## 评估

```bash
python evaluate.py --data-dir /path/to/FB15k --checkpoint runs/TransE/best.pt --split test
```

正式对比论文结果时应使用 `best.pt`；需要查看最近一轮时可以将 `--checkpoint` 改为 `last.pt`。可以通过 `--split valid` 在验证集上评估。快速检查流程时可以添加 `--limit 100`，但限制三元组数量后的结果不能作为完整测试结果。

## 提交到 GitHub

应提交源代码、`.gitignore`、`requirements.txt` 和 `README.md`。`.idea/`、`.venv/`、数据集、缓存以及训练生成的 `.pt` 文件均不应提交。
