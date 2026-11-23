# TransE

使用 PyTorch 实现的 TransE 知识图谱嵌入模型，支持在 FB15k 数据集上训练，并使用 Raw / Filtered Mean Rank 和 Hits@10 评估链接预测效果。仓库包含完整代码和 FB15k 数据，克隆后无需依赖额外的外层目录。

## 功能

- 从 FB15k 文本文件读取实体、关系和三元组
- 训练时随机替换头实体或尾实体进行负采样，并避免生成训练集中的正例
- 支持 L1、L2 距离以及 CPU、CUDA 训练
- 保存模型参数、优化器状态、训练配置和指标
- 支持头实体和尾实体预测，以及 Raw / Filtered MR 和 Hits@10

## 项目结构

```text
TransE/
├── .gitignore
├── README.md
├── requirements.txt
├── model.py
├── dataset.py
├── train.py
├── evaluate.py
└── data/
    └── FB15k/
        ├── entity2id.txt
        ├── relation2id.txt
        ├── train.txt
        ├── valid.txt
        └── test.txt
```

训练生成的检查点默认写入 `runs/TransE/`。该目录仅保留在本地，不提交到 Git。

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

## 数据集

本仓库包含 FB15k 的五个文本文件，默认位于 `data/FB15k/`。FB15k 是从 Freebase 构建的知识图谱数据集，常用于知识图谱嵌入和链接预测研究。

数据集来源及模型背景可参考 TransE 原始论文：

> Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko. Translating Embeddings for Modeling Multi-relational Data. NeurIPS 2013.

论文页面：<https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>

实体和关系映射文件使用制表符分隔名称与 ID；三元组文件依次包含头实体、尾实体和关系，同样使用制表符分隔。

检查数据能否正常加载：

```bash
python dataset.py
```

也可以通过 `--data-dir` 使用其他位置或格式的兼容数据集：

```bash
python dataset.py --data-dir /path/to/FB15k
```

## 训练

使用默认配置训练：

```bash
python train.py
```

程序默认读取 `data/FB15k/`，训练 1000 轮，嵌入维度为 50，批次大小为 1024，使用 L1 距离、1.0 的间隔和 0.01 的学习率。检查点保存为 `runs/TransE/last.pt`。

自定义训练参数：

```bash
python train.py --epochs 1000 --batch-size 1024 --embedding-dim 50 --distance l1 --learning-rate 0.01 --margin 1.0 --device auto
```

`--device` 可选 `auto`、`cpu` 或 `cuda`。`auto` 会在 CUDA 可用时使用 GPU，否则使用 CPU。可以通过 `--data-dir` 和 `--output-dir` 覆盖默认数据与输出路径。

## 评估

使用默认检查点在测试集上评估：

```bash
python evaluate.py
```

在验证集上评估：

```bash
python evaluate.py --split valid
```

可以通过 `--data-dir` 和 `--checkpoint` 指定其他数据集与检查点。快速检查流程时可以添加 `--limit 100`，但限制三元组数量后的结果不能作为完整测试结果。


