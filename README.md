# TransE

[简体中文](README.md) | **English**

A PyTorch implementation of the TransE knowledge graph embedding model. It supports training on the FB15k dataset and evaluating link prediction with Raw / Filtered Mean Rank (MR) and Hits@10.

## Features

- Load entities, relations, and triples from FB15k text files
- Perform negative sampling during training by randomly replacing head or tail entities while avoiding positive triples from the training set
- Support L1 and L2 distances and training on CPU or CUDA
- Save model parameters, optimizer state, training configuration, and metrics
- Evaluate periodically on the validation set, save the best model by Raw MR as `best.pt`, and retain the most recent model as `last.pt`
- Support head and tail entity prediction with Raw / Filtered MR and Hits@10

## Project Structure

```text
TransE/
├── .gitignore
├── model.py          # TransE model
├── dataset.py        # FB15k loading, negative sampling, and batch generation
├── train.py          # Training, validation, and checkpoint saving
├── evaluate.py       # Link prediction evaluation
├── requirements.txt  # Python dependencies
├── README.md         # Chinese documentation
├── README.en.md      # English documentation
└── data/
    └── FB15k/        # FB15k dataset
        ├── entity2id.txt
        ├── relation2id.txt
        ├── train.txt
        ├── valid.txt
        └── test.txt
```

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA version, install the corresponding build using the command provided on the [official PyTorch installation page](https://pytorch.org/get-started/locally/). Virtual environments and local PyTorch installation files should not be committed to Git.

## Data Preparation

The data directory must contain the following five files:

```text
FB15k/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── valid.txt
└── test.txt
```

Entity and relation mapping files contain names and IDs separated by tabs. Each triple file contains a head entity, a tail entity, and a relation, also separated by tabs.

The dataset is included in the repository under `data/FB15k/`. You can specify it with `--data-dir data/FB15k`. Alternatively, replace `/path/to/FB15k` with the path to another data directory.

You can check that the data loads correctly with:

```bash
python dataset.py --data-dir /path/to/FB15k
```

## Training

```bash
python train.py --data-dir /path/to/FB15k --output-dir runs/TransE
```

By default, the model trains for 1,000 epochs with an embedding dimension of 50, a batch size of 1024, L1 distance, a margin of 1.0, and a learning rate of 0.01. Backpropagation uses the batch-summed loss, consistent with the algorithm in the paper. The `loss` shown in the logs remains the average loss per triple for easier monitoring.

Evaluation runs every 10 epochs and after the final epoch on the first 1,000 validation triples. The best model is selected by validation Raw MR. Checkpoints are saved as:

- `runs/TransE/last.pt`: the model from the most recent epoch.
- `runs/TransE/best.pt`: the model with the lowest validation Raw MR.

Common options:

```bash
python train.py --data-dir /path/to/FB15k --output-dir runs/TransE --epochs 1000 --batch-size 1024 --embedding-dim 50 --distance l1 --learning-rate 0.01 --margin 1.0 --validation-interval 10 --validation-limit 1000
```

The program automatically selects a device based on the PyTorch environment of the current interpreter: it uses a GPU when CUDA is available and falls back to CPU otherwise. No additional device option is required. Set `--validation-limit` to `0` to use the full validation set, although each evaluation will take longer.

Reusing the same `--output-dir` overwrites its `last.pt` and `best.pt` files. Copy any checkpoints you want to keep before retraining. The `runs/` directory containing training artifacts is not committed to Git.

## Evaluation

```bash
python evaluate.py --data-dir /path/to/FB15k --checkpoint runs/TransE/best.pt --split test
```

Use `best.pt` for formal comparisons with results reported in papers. To evaluate the most recent epoch instead, change `--checkpoint` to `last.pt`. You can evaluate on the validation set with `--split valid`. For a quick workflow check, add `--limit 100`, but results from a limited number of triples are not valid full test results.
