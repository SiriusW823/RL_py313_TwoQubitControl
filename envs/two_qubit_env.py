import numpy as np

class TwoQubitEnv:
    """
    Two-qubit control environment.
    State is a 4-dimensional complex vector representing a quantum state.
    Actions use simple unitary rotations.
    """

    def __init__(self, max_steps=20, noise_sigma=0.0):
        self.max_steps = max_steps
        self.noise_sigma = noise_sigma
        self.reset()

    # ----------------------------------------------
    # Reset environment to |00>
    # ----------------------------------------------
    def reset(self):
        self.state = np.array([1, 0, 0, 0], dtype=np.complex128)
        self.steps = 0
        return self._get_obs()

    # ----------------------------------------------
    # Observation is real + imaginary parts
    # ----------------------------------------------
    def _get_obs(self):
        return np.concatenate([self.state.real, self.state.imag])  # shape = (8,)

    # ----------------------------------------------
    # Define 6 basic unitary operations
    # ----------------------------------------------
    def _get_action_unitary(self, action):
        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Tensor products
        IX = np.kron(I, X)
        XI = np.kron(X, I)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)

        # 6 available unitaries (example moves)
        ops = [
            np.eye(4, dtype=np.complex128),   # 0: identity
            IX,    # 1: X on qubit 2
            XI,    # 2: X on qubit 1
            IZ,    # 3: Z on qubit 2
            ZI,    # 4: Z on qubit 1
            -np.eye(4, dtype=np.complex128)   # 5: global phase (random)
        ]

        return ops[action]

    # --------------------------------------------------------
    # Add quantum noise AFTER the unitary operation
    # Depolarizing-like noise: ψ → ψ + σ ξ, renormalize
    # --------------------------------------------------------
    def add_quantum_noise(self, psi):
        if self.noise_sigma == 0:
            return psi

        sigma = self.noise_sigma

        # Complex Gaussian random vector ξ
        noise = (np.random.randn(*psi.shape) + 1j * np.random.randn(*psi.shape))
        noise /= np.linalg.norm(noise)

        noisy_psi = psi + sigma * noise
        noisy_psi /= np.linalg.norm(noisy_psi)  # renormalize

        return noisy_psi

    # ----------------------------------------------
    # Step function
    # ----------------------------------------------
    def step(self, action):
        self.steps += 1

        # Apply unitary
        U = self._get_action_unitary(action)
        psi = U @ self.state

        # Add physical noise
        psi = self.add_quantum_noise(psi)

        # Update state
        self.state = psi

        # Reward: fidelity to |00>
        target = np.array([1, 0, 0, 0], dtype=np.complex128)
        fidelity = np.abs(np.vdot(target, psi)) ** 2
        reward = float(fidelity)

        done = (self.steps >= self.max_steps)

        return self._get_obs(), reward, done, {}

