import numpy as np
import torch
import random

class UniformReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), n_step=1, gamma=0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) >= self.n_step:
            R, d = 0, False
            for i in range(self.n_step):
                r, dn = self.buffer[i][2], self.buffer[i][4]
                R += (self.gamma ** i) * r
                if dn:
                    d = True
                    break
            s, a = self.buffer[0][0], self.buffer[0][1]
            next_s = self.buffer[i][3]

            self.state[self.ptr] = s
            self.action[self.ptr] = a
            self.reward[self.ptr] = np.float32(R)
            self.next_state[self.ptr] = next_s
            self.done[self.ptr] = np.float32(d)

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            self.buffer.pop(0)

        if done:
            self.buffer.clear()

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[indices]),
            torch.FloatTensor(self.action[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.done[indices]),
        )

    def __len__(self):
        return self.size


class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree = np.zeros(2 * max_size - 1)
        self.data = np.zeros(max_size, dtype=object)
        self.write = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.write + self.max_size - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.max_size + 1
        return (idx, self.tree[idx], self.data[data_idx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000, alpha=0.6, beta=0.6, beta_increment=5e-4, eps=1e-6, n_step=1, gamma=0.99):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, priority=None):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) >= self.n_step:
            R, d = 0, False
            for i in range(self.n_step):
                r, dn = self.buffer[i][2], self.buffer[i][4]
                R += (self.gamma ** i) * r
                if dn:
                    d = True
                    break
            s, a = self.buffer[0][0], self.buffer[0][1]
            next_s = self.buffer[i][3]

            self.state[self.ptr] = s
            self.action[self.ptr] = a
            self.reward[self.ptr] = np.float32(R)
            self.next_state[self.ptr] = next_s
            self.done[self.ptr] = np.float32(d)


            if priority is None:
                priority = self.max_priority
            priority = np.clip(priority, a_min=1e-6, a_max=1.0)
            self.tree.add(priority, (s, a, R, next_s, d))
            self.priorities[self.ptr] = priority

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            self.buffer.pop(0)

        if done:
            self.buffer.clear()

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        self.beta = min(1.0, self.beta + self.beta_increment)
        probs = np.array(priorities) / self.tree.total()
        probs = np.clip(probs, 1e-6, 1.0)
        weights = (self.tree.size * probs + self.eps) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32).reshape(-1, 1)


        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(dones),
            idxs,
            torch.from_numpy(weights.astype(np.float32)).reshape(-1, 1)
        )


    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            priority = (np.abs(error) + self.eps) ** self.alpha
            data_idx = idx - self.tree.max_size + 1
            if 0 <= data_idx < self.max_size:
                self.priorities[data_idx] = priority
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size