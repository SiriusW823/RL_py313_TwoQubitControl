# RL_py313_TwoQubitControl

### Reinforcement Learning for Two-Qubit Quantum State Control (Python 3.13 Compatible)

---

## Abstract

This repository presents a fully Python 3.13–compatible framework for studying quantum-inspired reinforcement learning (RL) applied to two-qubit quantum state control.  
Two policy gradient RL algorithms—**A2C (Advantage Actor-Critic)** and **PPO (Proximal Policy Optimization)**—are benchmarked on a minimal two-qubit environment under:

1. **Clean (noise-free) conditions**, and  
2. **Noisy quantum dynamics** with Gaussian perturbations.

Results demonstrate that both algorithms achieve near-optimal control performance in the clean environment, while robustness under noise reveals convergence degradation patterns. PPO shows stronger stability relative to A2C under moderate noise.

All implementations avoid legacy `gym` dependencies, making this repository modern-compatible, lightweight, and academically reproducible.

---

## Key Features

- Full Python 3.13 compatibility (no `gym` required)
- Minimal two-qubit control environment
- Implementations of:
  - A2C (PyTorch)
  - PPO (with clipping objective)
- Configurable Gaussian noise modeling
- Reproducible RL benchmarking
- Built-in evaluation plots:
  - A2C vs PPO
  - Clean vs Noisy performance
- Clear experiment scripts with streamlined usage

---

# Mathematical Background

## 1. RL Setup

The agent interacts with the two-qubit state space via discrete action operators.

### State
Qubit amplitudes with state vector dimension:
\( s_t \in \mathbb{R}^8 \)

### Actions
Discrete set of 6 unitary-inspired transformations.

### Reward
Fidelity to target quantum state:

```math
R_t =
20 \cdot \left|\left\langle \psi_{\text{target}} \mid \psi_t \right\rangle\right|^2
````

Reward is normalized up to `20`.

---

## 2. A2C Policy Gradient Objective

```math
\nabla_\theta J(\theta) =
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\cdot A_t
\right]
```

Where the advantage is defined as:

```math
A_t = R_t - V_\phi(s_t)
```

---

## 3. PPO Clipped Objective

```math
L(\theta) =
\mathbb{E}
\left[
\min
\left(
r_t(\theta) \cdot A_t,
\;
\mathrm{clip}
\left(
r_t(\theta),
1 - \epsilon,
1 + \epsilon
\right)
\cdot A_t
\right)
\right]
```

Importance sampling ratio:

```math
r_t(\theta) =
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\text{old}}}(a_t \mid s_t)
}
```

Clipping constrains unbounded policy shifts and stabilizes learning.

---

# Project Structure

```text
RL_py313_TwoQubitControl/
│
├── algorithms/                         # RL core algorithms (PyTorch)
│   ├── a2c.py                           # A2C implementation
│   └── ppo.py                           # PPO implementation
│
├── envs/                                # Two-qubit simulations
│   ├── two_qubit_env.py                 # Clean environment
│   └── two_qubit_noisy_env.py           # Environment with Gaussian noise
│
├── experiments/                         # Training & analysis scripts
│   ├── train_a2c_two_qubit.py           # Train A2C on clean setup
│   ├── train_ppo_two_qubit.py           # Train PPO on clean setup
│   ├── compare_a2c_ppo.py               # A2C vs PPO (clean)
│   └── compare_clean_vs_noisy.py        # Clean vs noisy benchmarking
│
├── plots/                               # Generated figures
│   ├── a2c_clean_smooth.png
│   ├── ppo_clean_smooth.png
│   ├── clean_vs_noisy_sigma_0.05.png
│   └── a2c_vs_ppo_clean_smooth.png
│
├── requirements.txt                     # Python 3.13 dependencies
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

## 3. Generate all comparison plots

```bash
python experiments/compare_clean_vs_noisy.py
```

Plots will be output to:

```
plots/
```

---

# Results Overview

## Clean Environment

* A2C converges faster
* PPO is smoother and more stable
* Both reach near-optimal reward (≈ 20)

---

## Noisy Environment (σ = 0.05)

Increasing noise introduces:

* reward fluctuations
* slower convergence
* policy stability challenges

Findings:

* A2C retains slightly higher peak reward
* PPO exhibits greater long-term stability

Noise-sensitivity trends highlight the importance of:

* robust control policies
* noise-adaptive quantum learning designs

---

# Plots Included

## Clean:

* A2C (smoothed)
* PPO (smoothed)
* A2C vs PPO

## Clean vs Noisy:

* A2C (σ = 0.05)
* PPO (σ = 0.05)

All generated figures are saved under:

```
plots/
```

---

# License

MIT License.
Use freely with citation or reference.

---

# Citation (optional)

```
SiriusW823 (2025).
RL_py313_TwoQubitControl:
Reinforcement Learning for Two-Qubit Quantum State Control.
GitHub.
https://github.com/SiriusW823/RL_py313_TwoQubitControl
```

---
