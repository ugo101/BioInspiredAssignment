import gym
import numpy as np

# rescale from -1/1 to full action space
def rescale_action(action, low, high): # mathematically consistent with how action is squashed in SAC using tanH
    return low + 0.5 * (action + 1.0) * (high - low)

# wrapper for normalizing observations

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, epsilon=1e-8, clip_range=10.0):
        super(NormalizeObservation, self).__init__(env)
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(self.obs_shape, dtype=np.float32)
        self.running_var = np.ones(self.obs_shape, dtype=np.float32)
        self.count = epsilon

    def observation(self, obs):
        self._update_stats(obs)
        return self._normalize(obs)

    def _normalize(self, obs):
        normalized = (obs - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _update_stats(self, obs):
        batch_mean = obs
        batch_var = np.square(obs - self.running_mean)
        batch_count = 1

        total_count = self.count + batch_count
        delta = batch_mean - self.running_mean

        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.running_mean = new_mean
        self.running_var = new_var
        self.count = total_count

    def get_stats(self):
        return self.running_mean, self.running_var, self.count  # <-- This line is missing
    
    def load_stats(self, mean, var, count):
        self.running_mean = mean
        self.running_var = var
        self.count = count

class FixedNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, mean, var, epsilon=1e-8, clip_range=10.0):
        super(FixedNormalizeObservation, self).__init__(env)
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)

        assert self.mean.shape == env.observation_space.shape, "Mean shape must match observation space"
        assert self.var.shape == env.observation_space.shape, "Variance shape must match observation space"

    def observation(self, obs):
        return self._normalize(obs)

    def _normalize(self, obs):
        normalized = (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)

        return np.clip(normalized, -self.clip_range, self.clip_range)

    def get_stats(self):
        return self.mean, self.var, None  # For compatibility

    def load_stats(self, mean, var, _):
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)



def warmup_observation_normalization(env, num_episodes=10, max_steps=100):
    print(f"[INFO] Warming up observation normalization over {num_episodes} episodes...")
    for _ in range(num_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            # Take random actions just to sample state space
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    print(f"[INFO] Warm-up complete.")
