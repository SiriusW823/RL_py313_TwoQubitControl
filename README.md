# RL_py313_TwoQubitControl

### Reinforcement Learning for Two-Qubit Quantum State Control (Python 3.13 Compatible)

---

## Abstract

This repository presents a fully Python 3.13–compatible framework for studying quantum-inspired reinforcement learning (RL) applied to two-qubit state control.
The project benchmarks two policy gradient RL algorithms—**A2C (Advantage Actor-Critic)** and **PPO (Proximal Policy Optimization)**—on a simplified two-qubit environment under both:

1. **Clean (noise-free) conditions**, and
2. **Noisy quantum dynamics** with Gaussian perturbations.

Results demonstrate that both algorithms learn to achieve near-optimal reward in the clean environment, while robustness under noise reveals convergence degradation patterns. PPO shows stronger stability under moderate noise relative to A2C.

All implementations avoid legacy `gym` dependencies, making this repository modern-compatible, lightweight, and suitable for academic reproducibility.

---

## Key Features

* Python 3.13 compatible (no gym dependency)
* Minimalistic two-qubit control environment
* Implementations of:
  * A2C (PyTorch)
  * PPO with clipping objective
* Support for noise-perturbed quantum dynamics
* Reproducible RL experiments
* Plots comparing:
  * A2C vs PPO
  * Clean vs Noisy reward trends
* Clear experiment scripts with streamlined usage

---

# Mathematical Background

### 1. Reinforcement Learning Setup

The agent interacts with the two-qubit state space via discrete action operators.

- State: qubit amplitudes, with the state vector dimension
  $s_t \in \mathbb{R}^{8}$.

- Actions: basic unitary gates (mapped to 6 discrete operations).

- Reward: fidelity to target quantum state
  $$
  R_t = 20 \cdot \left|\left\langle \psi_{\text{target}} \mid \psi_t \right\rangle\right|^2,
  $$
  normalized up to `20`.

---

### 2. A2C Policy Update

The policy gradient objective for A2C is
$$
\nabla_\theta J(\theta) =
\mathbb{E}\!\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t
\right],
$$

where the advantage is
$$
A_t = R_t - V_\phi(s_t).
$$

---

### 3. PPO Clipped Objective

PPO maximizes the clipped surrogate objective:
$$
L(\theta) =
\mathbb{E}\!\left[
\min\!\left(
r_t(\theta)\cdot A_t,\;
\mathrm{clip}\!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\cdot A_t
\right)
\right],
$$

with the importance ratio
$$
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

Clipping prevents unstable large policy updates.

---

## Project Structure

```text
RL_py313_TwoQubitControl/
│
├── algorithms/                         # RL core algorithms (PyTorch)
│   ├── a2c.py                           # Advantage Actor-Critic implementation
│   └── ppo.py                           # Proximal Policy Optimization implementation
│
├── envs/                                # Two-qubit simulators
│   ├── two_qubit_env.py                 # Clean environment (noise-free)
│   └── two_qubit_noisy_env.py           # Noisy environment with configurable Gaussian noise
│
├── experiments/                         # Training & evaluation scripts
│   ├── train_a2c_two_qubit.py           # Train A2C on clean environment
│   ├── train_ppo_two_qubit.py           # Train PPO on clean environment
│   ├── compare_a2c_ppo.py               # Compare A2C vs PPO performance (clean)
│   └── compare_clean_vs_noisy.py        # Compare clean vs noisy env (A2C & PPO)
│
├── plots/                               # Auto-generated experiment figures
│   ├── a2c_clean_smooth.png
│   ├── ppo_clean_smooth.png
│   ├── clean_vs_noisy_sigma_0.05.png
│   └── a2c_vs_ppo_clean_smooth.png
│
├── requirements.txt                     # All dependencies for Python 3.13
├── README.md
├── LICENSE
├── .gitignore
└── .gitattributes
```

---

# Installation

## 1. Create Virtual Environment (Recommended)

### Windows PowerShell

```bash
python -m venv rl313
rl313\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv rl313
source rl313/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Experiments

## 1. Train A2C (clean environment)

```bash
python experiments/train_a2c_two_qubit.py
```

## 2. Train PPO (clean environment)

```bash
python experiments/train_ppo_two_qubit.py
```

## 3. Generate comparison plots

(A2C vs PPO clean/noisy)

```bash
python experiments/compare_clean_vs_noisy.py
```

Outputs will be saved in:

```
plots/
```

---

# Results Overview

## Clean Environment

* A2C converges faster and reaches maximum reward earlier.
* PPO converges more smoothly but slower in initial learning.

Both methods:

> achieve near-optimal control performance  
> (max reward ≈ 20)

---

## Noisy Environment (σ = 0.05)

* Both algorithms suffer degraded learning stability
* Reward volatility increases
* A2C retains slightly better peak performance
* PPO shows smoother averaged convergence

Quantum-noise sensitivity becomes a key research signal for:

* robust control models
* noise-aware policy design

---

# Plots Included

### Clean:

* A2C (smoothed)
* PPO (smoothed)
* A2C vs PPO

### Clean vs Noisy:

* A2C (σ = 0.05)
* PPO (σ = 0.05)

All figures are saved under:

```
plots/
```

---

# License

MIT License.
Use freely with citation or reference.

---

# Citation (optional)

If used in academic reports:

```
SiriusW823 (2025). RL_py313_TwoQubitControl:
Reinforcement Learning for Two-Qubit Quantum State Control, GitHub.
https://github.com/SiriusW823/RL_py313_TwoQubitControl
```