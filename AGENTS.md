# AGENTS.md — VPC-Labs

## Setup
- Use `conda` env `VPC_env` with Python 3.13 + `uv` for packages (see README).
- `pip install -e .` installs `src/` and `scripts/` as top-level importable packages, not `app.*`. All internal imports use `from src.xxx import ...` and `from scripts import ...`.

## Entrypoint
```bash
python app/main.py gender [args]
python app/main.py car [args]
```
Run from repo root. The CLI uses argparse subcommands with `--seed`, `--batch_size`, `--epochs`, `--learning_rate`, etc., plus subcommand-specific flags like `--model_type`.

## Architecture
- `app/main.py` — CLI, maps subcommands to `cmd_gender`/`cmd_car`
- `app/scripts/gender.py`, `app/scripts/car.py` — training loops (PyTorch Lightning)
- `app/src/config/config.py` — `Configuration` dataclass; all hyperparams live here
- `app/src/data/gender.py`, `app/src/data/car.py` — data loading from `.npy` files
- `app/src/models/gender.py` — `GenderCNN`, `GenderCNNSmall`, `GenderResNet`, `GenderModule`
- `app/src/models/car.py` — `BilinearCarCNN` (VGG/ResNet bilinear), `CarModule`

## Data prerequisites
- Data must exist as `.npy` files at hardcoded paths: `data/gender/x_train.npy`, `data/gender/y_train.npy`, etc., and `data/car/x_train.npy`, etc.
- `Configuration.__post_init__` auto-creates `models/`, `logs/`, and subdirectories on startup.

## Training quirks
- **Gender**: single-stage training. `--model_type` choices: `small`, `large`, `resnet`.
- **Car**: two-stage training (freeze backbones → unfreeze + lower LR). `--model_type` choices: `vg-res`, `vg-vg`, `res-res`.
- Both use `AdamW` + `CosineAnnealingWarmRestarts` by default. Early stopping monitors `val_acc`.
- `Configuration` defaults differ from CLI arg defaults (e.g., `--epochs` CLI default is 100, config default is 100 — same, but `--patience` CLI default 20 vs config default 10). CLI wins via `args_to_dataclass`.

## Key dependencies
`pytorch-lightning`, `torch`, `torchvision`, `numpy`, `maikol-utils` (custom helpers: `print_separator`, `args_to_dataclass`, `make_dirs`)

## No tooling
- No test suite, no linter/formatter config, no CI, no type checker. This is a research/lab project.
- `.vscode/` is gitignored.
- `data/`, `logs/`, `models/` contents are gitignored (only `.gitkeep` preserved).
