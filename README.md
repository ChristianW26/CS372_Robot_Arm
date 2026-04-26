## CS372 Robot Arm — PPO for ManiSkill PickCube (Simulation)

### Project goal 

Train a **PPO** policy in simulation for the **ManiSkill `PickCube-v1` manipulation task** with the `so100` robot (state observations + joint-delta control) and evaluate whether training improves task performance over time.

### Demo video (non-technical)
This section is meant to satisfy: **“Project demo video effectively communicates why the project matters to a non-technical audience in non-technical terms.”**

- **Main demo link (recommended)**: *TODO: add YouTube/Drive link (best for inline playback)*
- **Training progression clips (MP4s in this repo)**:
  - **Preliminary**: [`media/preliminary.mp4`](./media/preliminary.mp4) — moves near the cube but struggles to grasp it
  - **In progress**: [`media/Inprogress.mp4`](./media/Inprogress.mp4) — grasps the cube but struggles to reach/complete the goal
  - **Final**: [`media/final.mp4`](./media/final.mp4) — completes the full task (grasp + goal)


### Why this is meaningful 

Robot manipulation is a core research problem, and simulation benchmarks let us test learning-based control methods reproducibly. This project is grounded in:

- **Benchmark environment**: ManiSkill’s manipulation tasks (we use `PickCube-v1` via `gymnasium` + `mani-skill`).  
  - ManiSkill: a widely used robotics RL benchmark suite for learning-based manipulation in simulation.
- **RL algorithm**: **Proximal Policy Optimization (PPO)** (Schulman et al.), a standard baseline in modern continuous-control RL.
- **Research question**: does PPO training measurably improve success and efficiency on a standardized pick-and-lift task (and what reward shaping helps)?

### Technical walkthrough 

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
  - Training logs are written to `**Training Data.csv`** with:
    - `Actor Losses`, `Critic Losses`, `Total Rewards`, `Avg Lengths`
- **(3) Offline evaluation visualization** (`ppo.ipynb`)
  - Plots smoothed curves for the above metrics and renders an evaluation rollout video in-notebook.

### Problem → approach → solution → evaluation 

- **Problem**: Hand-tuning a controller for pick-and-lift is brittle; we want to learn it from interaction.
- **Approach**: Use ManiSkill `PickCube-v1` as a standardized manipulation benchmark and train PPO for a fixed number of iterations.
- **Solution**: A trained PPO actor/critic (saved as `.pth`) and a repeatable notebook that logs metrics every training run.
- **Evaluation**: Use objective learning curves (losses + rewards + episode length) and a short demo video that shows qualitative behavior.

### Evaluation metrics 

The objective in `PickCube-v1` is to complete the pick-and-lift task reliably and efficiently. The code already measures:

- **Success / completion signal**: the environment reward reaching `1.0` (used in training as a trigger for `success_bonus`).
- **Average episode length** (`Avg Lengths`): shorter is better given a positive success objective (efficiency).
- **Total reward** (`Total Rewards`): includes base environment reward + success bonus − time penalty, so it is aligned with “succeed quickly”.
- **Actor/Critic losses**: training stability diagnostics (not a task metric, but used to verify learning behavior).
- **Iterations / batches**: the training loop runs for a fixed number of batches (the notebook uses `num_batches=1_000`), so plots can be interpreted as “metric vs iteration”.
- ***Maybe add success rate over N evaluations metric***

### Training progression (what changed and why)
Across the three clips in `media/`, performance improves in a way that matches the plotted metrics:

- **Preliminary**: the policy tends to approach the cube but fails to consistently grasp.
- **In progress**: the policy learns to grasp but is inefficient/inconsistent about reaching the goal state.
- **Final**: the policy completes the full sequence.

The biggest improvements came from:
- **Reward shaping**:
  - adding a **negative reward per timestep** (time penalty) to push faster completion
  - adding a **large reward for completing the task** (success bonus) to strongly reinforce successful episodes
- **More training experience**: training for **more episodes / iterations** (more batches collected from the environment).

### Design choices

These are the key implementation choices and why they exist:

- `**obs_mode="state"`**: focuses on learning control dynamics without adding the complexity of vision-based learning.
- `**control_mode="pd_joint_delta_pos"**`: provides a stable continuous-control interface for PPO (small joint updates each step).
- **Reward shaping**:
  - **Success bonus** encourages task completion rather than “hovering near success”.
  - **Time penalty** discourages long episodes and aligns with “lift quickly”.

### Repo layout

- `**ppo.ipynb`**: training + plotting + sim evaluation for PPO on ManiSkill `PickCube-v1` with `so100`.
- **Artifacts**:
  - `**Training Data.csv`**: training log output from `ppo.ipynb`.
  - `**ppo_actor.pth`, `ppo_critic.pth**`: saved model weights from `ppo.ipynb` training.

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

