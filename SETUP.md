## Setup

### 1) Prerequisites

- uv python package manager (available [here](https://docs.astral.sh/uv/getting-started/installation/))
- **Python**: 3.12+ (matches `pyproject.toml`)
- A working C/C++ toolchain (usually already present on macOS/Linux)

### 2) Install dependencies and create virtual environment

#### Option A — `uv` (recommended)

```bash
uv sync
```

#### Option B — pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run the project

#### Launch Jupyter

```bash
jupyter lab
```

#### Run training + evaluation

Open and run:

- `notebooks/ppo_training.ipynb`

### 4) Where outputs go

- **Training Logs**: `models/ppo_training_data.csv`
- **Model Parameters**: `models/ppo_actor.pth`, `models/ppo_critic.pth`
- **Demo clips**: `videos/`

### Troubleshooting

- If you see GPU-related issues: the notebook currently checks for CUDA availability; either run on a machine with CUDA or adjust the notebook to allow CPU training.

