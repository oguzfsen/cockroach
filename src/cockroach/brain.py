"""Small, independent Double DQN brains, implemented entirely with NumPy.

No teacher policy or food coordinates. Replay, a target network, Huber loss,
gradient clipping and Adam are intentionally explicit and inspectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cockroach.world import OBS_SIZE, Action

EXPLORATION = np.array([0.43, 0.16, 0.16, 0.16, 0.04, 0.04, 0.01])


@dataclass(frozen=True)
class BrainConfig:
    hidden: int = 48
    replay_size: int = 30_000
    batch_size: int = 64
    warmup: int = 256
    train_every: int = 4
    target_every: int = 200
    learning_rate: float = 0.0007
    gamma: float = 0.985
    epsilon_min: float = 0.06
    epsilon_decay: float = 20_000.0


class Network:
    def __init__(self, rng: np.random.Generator, hidden: int):
        dims = [OBS_SIZE, hidden, hidden, len(Action)]
        self.params = {}
        for i, (inputs, outputs) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            self.params[f"w{i}"] = (
                rng.standard_normal((inputs, outputs)) * np.sqrt(2.0 / inputs)
            ).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(outputs, dtype=np.float32)

    def forward(self, x: np.ndarray, cache: bool = False):
        activations = [x]
        for i in range(3):
            x = x @ self.params[f"w{i}"] + self.params[f"b{i}"]
            if i < 2:
                x = np.maximum(x, 0)
            activations.append(x)
        return (x, activations) if cache else x

    def gradients(self, states, actions, targets):
        q, activations = self.forward(states, cache=True)
        batch = len(states)
        errors = q[np.arange(batch), actions] - targets
        loss = np.where(np.abs(errors) < 1, 0.5 * errors**2, np.abs(errors) - 0.5).mean()
        grad = np.zeros_like(q)
        grad[np.arange(batch), actions] = np.clip(errors, -1, 1) / batch
        grads = {}
        for i in (2, 1, 0):
            grads[f"w{i}"] = activations[i].T @ grad
            grads[f"b{i}"] = grad.sum(axis=0)
            if i:
                grad = (grad @ self.params[f"w{i}"].T) * (activations[i] > 0)
        return float(loss), grads

    def copy_from(self, other: Network):
        for key in self.params:
            self.params[key][:] = other.params[key]


class Replay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = np.empty((capacity, OBS_SIZE), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(self, state, action, reward, next_state, done):
        i = self.cursor
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, count: int, rng: np.random.Generator):
        indices = rng.integers(self.size, size=count)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class Brain:
    def __init__(self, seed: int, config: BrainConfig | None = None, training: bool = True):
        self.config = config or BrainConfig()
        c = self.config
        self.rng = np.random.default_rng(seed)
        self.network = Network(self.rng, c.hidden)
        self.target = Network(self.rng, c.hidden)
        self.target.copy_from(self.network)
        self.replay = Replay(c.replay_size) if training else None
        self.steps = 0
        self.updates = 0
        self.last_loss = 0.0
        self.m = {key: np.zeros_like(value) for key, value in self.network.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.network.params.items()}

    @property
    def epsilon(self) -> float:
        c = self.config
        return c.epsilon_min + (1 - c.epsilon_min) * np.exp(-self.steps / c.epsilon_decay)

    def act(self, state: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(len(Action), p=EXPLORATION))
        return int(self.network.forward(state).argmax())

    def remember(self, state, action, reward, next_state, done):
        if self.replay is None:
            raise RuntimeError("Gözlemci modundaki ağ öğrenemez.")
        self.replay.add(state, action, reward, next_state, done)
        self.steps += 1
        c = self.config
        if self.replay.size >= max(c.warmup, c.batch_size) and self.steps % c.train_every == 0:
            self.learn()

    def enable_learning(self):
        """Enable replay without resetting learned weights, Adam, RNG or counters."""
        if self.replay is None:
            self.replay = Replay(self.config.replay_size)

    def learn(self):
        c = self.config
        states, actions, rewards, next_states, dones = self.replay.sample(c.batch_size, self.rng)
        # Double DQN: online network chooses; the delayed target network evaluates.
        choices = self.network.forward(next_states).argmax(axis=1)
        next_q = self.target.forward(next_states)[np.arange(c.batch_size), choices]
        targets = rewards + c.gamma * (1.0 - dones) * next_q
        loss, grads = self.network.gradients(states, actions, targets)
        self.last_loss = loss
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = float(min(1.0, 5.0 / (norm + 1e-8)))
        self.updates += 1
        for key, grad in grads.items():
            grad = grad * scale
            self.m[key] *= 0.9
            self.m[key] += 0.1 * grad
            self.v[key] *= 0.999
            self.v[key] += 0.001 * grad * grad
            m_hat = self.m[key] / (1 - 0.9**self.updates)
            v_hat = self.v[key] / (1 - 0.999**self.updates)
            self.network.params[key] -= c.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        if self.updates % c.target_every == 0:
            self.target.copy_from(self.network)

    def metadata(self) -> dict:
        return {
            "config": asdict(self.config),
            "steps": self.steps,
            "updates": self.updates,
            "last_loss": self.last_loss,
            "rng": self.rng.bit_generator.state,
        }

    def arrays(self, prefix: str) -> dict:
        arrays = {}
        for name, mapping in (
            ("online", self.network.params),
            ("target", self.target.params),
            ("m", self.m),
            ("v", self.v),
        ):
            for key, value in mapping.items():
                arrays[f"{prefix}_{name}_{key}"] = value
        return arrays

    def restore(self, metadata: dict, arrays, prefix: str):
        for name, mapping in (
            ("online", self.network.params),
            ("target", self.target.params),
            ("m", self.m),
            ("v", self.v),
        ):
            for key, value in mapping.items():
                data = arrays[f"{prefix}_{name}_{key}"]
                if data.shape != value.shape or not np.isfinite(data).all():
                    raise ValueError(f"Bozuk ağ ağırlığı: {prefix}/{name}/{key}")
                value[:] = data
        self.steps = metadata["steps"]
        self.updates = metadata["updates"]
        self.last_loss = metadata["last_loss"]
        self.rng.bit_generator.state = metadata["rng"]
