# VPC Labs — Computer Vision Exercises

[English](#english) | [Español](#español)

---

<a name="english"></a>
## English

### About
This repository contains the three assignments for the **Computer Vision** course (MUIARFID):
1. **Gender Recognition** — Train a CNN (<100K or >98% accuracy)
2. **Car Model Identification** — Bilinear CNN with pre-trained backbones (ResNet-50 / VGG-16)
3. **Attention Visualization** — ViT attention maps, rollout, and interactive per-patch exploration

**Repository Structure:**

* `app/main.py` — CLI entry point for training (gender / car subcommands)
* `app/scripts/` — Training loops (`gender.py`, `car.py`) using PyTorch Lightning
* `app/src/config/` — `Configuration` dataclass with all hyperparameters
* `app/src/data/` — Data loaders from `.npy` files
* `app/src/models/` — Model architectures (CNN, ResNet, Bilinear CNN)
* `data/` — Datasets (`.npy` format: gender, car, attention/validation images)
* `notebooks/` — 4 notebooks for each exercise + interactive attention visualization
* `models/` — Saved checkpoints (ignored by git)
* `logs/` — Training logs (ignored by git)
* `docs/` — Project documentation

### Setup & Installation

```bash
conda create --name VPC_env python=3.13 -y
conda activate VPC_env

pip install uv
uv pip install -r requirements.txt
pip install -e .
uv pip install ipykernel
python -m ipykernel install --user --name=VPC_env --display-name "VPC_env (Conda)"
```

### Execution

**Training via CLI (from repo root):**

```bash
# Gender Recognition
python app/main.py gender --model_type resnet --epochs 100 --batch_size 128 --learning_rate 5e-3

# Car Model Identification  
python app/main.py car --model_type vg-res --epochs 100 --batch_size 128 --learning_rate 5e-3
```

Common flags: `--seed`, `--batch_size`, `--epochs`, `--learning_rate`, `--weight_decay`, `--label_smoothing`, `--patience`.

Gender-specific: `--model_type` (`small`, `large`, `resnet`), `--image_size`, `--dropout_rate`.

Car-specific: `--model_type` (`vg-res`, `vg-vg`, `res-res`), `--momentum`.

**Notebooks (from repo root):**
```bash
jupyter notebook notebooks/
```

| Notebook | Exercise |
|---|---|
| `gender_prediction.ipynb` | Gender Recognition (CNN) |
| `car_model_identification.ipynb` | Car Model Identification (Bilinear CNN) |
| `attention_visualization.ipynb` | ViT attention maps & rollout |
| `attention_visualization_interactive.ipynb` | Interactive per-patch attention explorer |

All notebooks import from `src.*` and `scripts.*` directly (no path changes needed).

### Data
Data must be placed as `.npy` files under `data/gender/` and `data/car/` before training. Validation images for attention visualization go under `data/attention/validation/`.

### Key Dependencies
`pytorch-lightning`, `torch`, `torchvision`, `numpy`, `maikol-utils`, `transformers`, `timm`, `ipykernel`.

### Dataset Source
Provided by the course. Place `.npy` files in the corresponding `data/` subdirectories.

*Maintained by [MiquelGomezCorral](https://miquelgc.net)*

---

<a name="español"></a>
## Español

### Sobre el proyecto
Este repositorio contiene las tres prácticas de la asignatura de **Visión por Computador** (MUIARFID):
1. **Gender Recognition** — Entrenar una CNN (<100K parámetros o >98% accuracy)
2. **Car Model Identification** — CNN bilineal con backbones preentrenados (ResNet-50 / VGG-16)
3. **Attention Visualization** — Mapas de atención de ViT, rollout y exploración interactiva por parche

**Estructura del Repositorio:**

* `app/main.py` — Punto de entrada CLI para entrenamiento (subcomandos gender / car)
* `app/scripts/` — Bucles de entrenamiento (`gender.py`, `car.py`) con PyTorch Lightning
* `app/src/config/` — Dataclass `Configuration` con todos los hiperparámetros
* `app/src/data/` — Carga de datos desde archivos `.npy`
* `app/src/models/` — Arquitecturas (CNN, ResNet, CNN Bilineal)
* `data/` — Datasets (formato `.npy`: gender, car, imágenes de validación para attention)
* `notebooks/` — 4 notebooks para cada ejercicio + visualización interactiva de atención
* `models/` — Checkpoints guardados (ignorados por git)
* `logs/` — Logs de entrenamiento (ignorados por git)
* `docs/` — Documentación del proyecto

### Configuración e Instalación

```bash
conda create --name VPC_env python=3.13 -y
conda activate VPC_env

pip install uv
uv pip install -r requirements.txt
pip install -e .
uv pip install ipykernel
python -m ipykernel install --user --name=VPC_env --display-name "VPC_env (Conda)"
```

### Ejecución

**Entrenamiento por CLI (desde la raíz del repo):**

```bash
# Gender Recognition
python app/main.py gender --model_type resnet --epochs 100 --batch_size 128 --learning_rate 5e-3

# Car Model Identification  
python app/main.py car --model_type vg-res --epochs 100 --batch_size 128 --learning_rate 5e-3
```

Flags comunes: `--seed`, `--batch_size`, `--epochs`, `--learning_rate`, `--weight_decay`, `--label_smoothing`, `--patience`.

Específicos de gender: `--model_type` (`small`, `large`, `resnet`), `--image_size`, `--dropout_rate`.

Específicos de car: `--model_type` (`vg-res`, `vg-vg`, `res-res`), `--momentum`.

**Notebooks (desde la raíz del repo):**
```bash
jupyter notebook notebooks/
```

| Notebook | Ejercicio |
|---|---|
| `gender_prediction.ipynb` | Gender Recognition (CNN) |
| `car_model_identification.ipynb` | Car Model Identification (CNN Bilineal) |
| `attention_visualization.ipynb` | Mapas de atención ViT y rollout |
| `attention_visualization_interactive.ipynb` | Explorador interactivo de atención por parche |

Todos los notebooks importan desde `src.*` y `scripts.*` directamente (no hace falta modificar el path).

### Datos
Los datos deben colocarse como archivos `.npy` en `data/gender/` y `data/car/` antes de entrenar. Las imágenes de validación para la visualización de atención van en `data/attention/validation/`.

### Dependencias principales
`pytorch-lightning`, `torch`, `torchvision`, `numpy`, `maikol-utils`, `transformers`, `timm`, `ipykernel`.

### Fuente del dataset
Proporcionado por la asignatura. Colocar los archivos `.npy` en los subdirectorios correspondientes de `data/`.

*Mantenido por [MiquelGomezCorral](https://miquelgc.net)*
