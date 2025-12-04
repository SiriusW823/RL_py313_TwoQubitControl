# RL_py313 Quantum RL Skeleton

This project is a Python 3.13–compatible skeleton for quantum-inspired RL
experiments with two-qubit environments (noise-free) and A2C/PPO algorithms.

## Structure

- `envs/two_qubit_env.py` – simple two-qubit toy env, no noise.
- `algorithms/a2c.py` – basic A2C implementation (PyTorch + gymnasium).
- `algorithms/ppo.py` – basic clipped PPO implementation.
- `experiments/train_a2c_two_qubit.py` – run A2C on the env.
- `experiments/train_ppo_two_qubit.py` – run PPO on the env.

## Setup

```bash
python -m venv rl313
source rl313/bin/activate  # or rl313\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run

```bash
python experiments/train_a2c_two_qubit.py
python experiments/train_ppo_two_qubit.py
```
