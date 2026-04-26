## Setup

### 1) Prerequisites

- **Python**: 3.12+ (matches `pyproject.toml`)
- A working C/C++ toolchain (usually already present on macOS/Linux)

### 2) Install dependencies

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

- `rl/ppo.ipynb`

### 4) Where outputs go

- **Training logs**: `rl/Training Data.csv`
- **Weights**: `rl/ppo_actor.pth`, `rl/ppo_critic.pth`
- **Demo clips**: `media/`

### Troubleshooting

- If you see GPU-related issues: the notebook currently checks for CUDA availability; either run on a machine with CUDA or adjust the notebook to allow CPU training.

