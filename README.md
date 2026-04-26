## CS372 Robot Arm — PPO for ManiSkill PickCube (Simulation)

### Project goal (single unified objective)
Train a **PPO** policy in simulation for the **ManiSkill `PickCube-v1` manipulation task** with the `so100` robot (state observations + joint-delta control) and evaluate whether training improves task performance over time.

### Why this is meaningful (real-world + research grounding)
Robot manipulation is a core research problem, and simulation benchmarks let us test learning-based control methods reproducibly. This project is grounded in:

- **Benchmark environment**: ManiSkill’s manipulation tasks (we use `PickCube-v1` via `gymnasium` + `mani-skill`).  
  - ManiSkill: a widely used robotics RL benchmark suite for learning-based manipulation in simulation.
- **RL algorithm**: **Proximal Policy Optimization (PPO)** (Schulman et al.), a standard baseline in modern continuous-control RL.
- **Research question**: does PPO training measurably improve success and efficiency on a standardized pick-and-lift task (and what reward shaping helps)?

### Technical walkthrough (how the components fit together)
This repo is a single pipeline, not isolated experiments:

- **(1) Simulation environment** (`ppo.ipynb`)
  - Creates the task with `gym.make("PickCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos", robot_uids="so100")`
  - Wraps it with `ManiSkillVectorEnv` for batched rollout collection.

- **(2) PPO training loop** (`ppo.ipynb`)
  - Actor/Critic MLPs are trained on batches of trajectories.
  - Training runs for a configured number of **iterations** (the notebook uses `num_batches=1_000`).
  - Rewards are shaped by two explicit design choices already implemented:
    - **success bonus**: `reward += (reward == 1.0) * success_bonus`
    - **time penalty**: `reward -= time_penalty`
  - Training logs are written to **`Training Data.csv`** with:
    - `Actor Losses`, `Critic Losses`, `Total Rewards`, `Avg Lengths`

- **(3) Offline evaluation visualization** (`ppo.ipynb`)
  - Plots smoothed curves for the above metrics and renders an evaluation rollout video in-notebook.

### Problem → approach → solution → evaluation (explicit progression)

- **Problem**: Hand-tuning a controller for pick-and-lift is brittle; we want to learn it from interaction.
- **Approach**: Use ManiSkill `PickCube-v1` as a standardized manipulation benchmark and train PPO for a fixed number of iterations.
- **Solution**: A trained PPO actor/critic (saved as `.pth`) and a repeatable notebook that logs metrics every training run.
- **Evaluation**: Use objective learning curves (losses + rewards + episode length) and a short demo video that shows qualitative behavior.

### Evaluation metrics (tied directly to the objective)
The objective in `PickCube-v1` is to complete the pick-and-lift task reliably and efficiently. The code already measures:

- **Success / completion signal**: the environment reward reaching `1.0` (used in training as a trigger for `success_bonus`).
- **Average episode length** (`Avg Lengths`): shorter is better given a positive success objective (efficiency).
- **Total reward** (`Total Rewards`): includes base environment reward + success bonus − time penalty, so it is aligned with “succeed quickly”.
- **Actor/Critic losses**: training stability diagnostics (not a task metric, but used to verify learning behavior).
- **Iterations / batches**: the training loop runs for a fixed number of batches (the notebook uses `num_batches=1_000`), so plots can be interpreted as “metric vs iteration”.

If you want a single headline number, add a **success rate over N evaluation episodes** (this repo already contains the needed success signal; it’s the natural “objective metric” for pick-and-lift).

### Project demo video (non-technical audience)
Add a short video link here that explains the “why” in plain language (no RL jargon), and shows the behavior:

- **Video link**: _TODO: add YouTube/Drive link_


### Design choices (documented + justified)
These are the key implementation choices and why they exist:

- **`obs_mode="state"`**: focuses on learning control dynamics without adding the complexity of vision-based learning.
- **`control_mode="pd_joint_delta_pos"`**: provides a stable continuous-control interface for PPO (small joint updates each step).
- **Reward shaping**:
  - **Success bonus** encourages task completion rather than “hovering near success”.
  - **Time penalty** discourages long episodes and aligns with “lift quickly”.

### Repo layout (what each file is for)

- **`ppo.ipynb`**: training + plotting + sim evaluation for PPO on ManiSkill `PickCube-v1` with `so100`.
- **Artifacts**:
  - **`Training Data.csv`**: training log output from `ppo.ipynb`.
  - **`ppo_actor.pth`, `ppo_critic.pth`**: saved model weights from `ppo.ipynb` training.

### Setup
This project uses **Python ≥ 3.12** (see `pyproject.toml`) and depends on `gymnasium`, `mani-skill`, `torch`, and plotting/logging libraries.

If you use `uv`:

```bash
uv sync
```

Or with pip (conceptually):

```bash
pip install -e .
```

### Run: training + evaluation in simulation
Open and run:

- `ppo.ipynb`

Key configuration (from the notebook):
- `env_id`: `PickCube-v1`
- `obs_mode`: `state`
- `control_mode`: `pd_joint_delta_pos`
- `robot_uids`: `so100`
- vectorized rollout: `num_envs = 128`
- training iterations: `num_batches = 1_000`


