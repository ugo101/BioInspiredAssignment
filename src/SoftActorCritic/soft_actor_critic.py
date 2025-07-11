import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.stats
from torch.utils.tensorboard import SummaryWriter

path = '/home/damenadmin/Projects/SoftActorCritic/src'
sys.path.append(path)

from SoftActorCritic.actor_network import ActorNetwork
from SoftActorCritic.critic_network import QNetwork
from SoftActorCritic.target_network import soft_update
from SoftActorCritic.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_layers_actor=[256, 256],
        hidden_layers_critic=[256, 256],
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy=None,
        device=None,
        log_path=None,  
    ):
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")

        # === Actor ===
        self.actor = ActorNetwork(state_dim, action_dim, hidden_layers_actor).to(self.device) # actions = rpm_ps, alpha_ps, rpm_sb, alpha_sb
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # With L2 regularization:
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)  # adjust 1e-4 as needed


        # === Critics ===
        self.q1 = QNetwork(state_dim, action_dim, hidden_layers_critic).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_layers_critic).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_layers_critic).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_layers_critic).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        # self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # With L2 regularization:
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr, weight_decay=5e-4) # increase weight due to large truncation reward such to not overfit
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr, weight_decay=5e-4)

        # === Entropy (α) tuning ===
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.log_alpha = torch.tensor(np.log(0.05), requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Logging
        if log_path is not None:
            self.writer = SummaryWriter(log_dir=log_path + f"/SAC_{int(time.time())}")
        else:
            # Create a new directory for TensorBoard logs
            os.makedirs("runs", exist_ok=True)
            # Use current time for unique log directory
            self.writer = SummaryWriter(log_dir=f"runs/SAC_{int(time.time())}")

        self.total_train_steps = 0
        self.total_episodes = 0

        self.replay_buffer = None # used for saving replay buffer when saving the model

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, mean, _, _, _ = self.actor(state)
        return mean.detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size, use_prioritized=False):
        if use_prioritized:
            states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(batch_size)
            dones = dones.float()
        else:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            dones = dones.float()
            weights = torch.ones_like(rewards)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # --- Target Q ---
        with torch.no_grad():
            next_actions, next_log_probs, _, _, _, _ = self.actor(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * (min_q_next - self.alpha.detach() * next_log_probs)

        # --- Critic losses ---
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = (F.mse_loss(q1_pred, target_q, reduction='none') * weights).mean()
        q2_loss = (F.mse_loss(q2_pred, target_q, reduction='none') * weights).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- Actor loss ---
        new_actions, log_probs, _, mean, log_std, u = self.actor(states)
        std = log_std.exp()

        # print(log_probs)
        q1_val = self.q1(states, new_actions)
        q2_val = self.q2(states, new_actions)
        min_q = torch.min(q1_val, q2_val)
        actor_loss = (self.alpha.detach() * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha tuning ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft updates ---
        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)

        # # check if q values not exploding
        # print("Target Q mean:", target_q.mean().item())

        # # check reward
        # print("Reward mean/std:", rewards.mean().item(), rewards.std().item())



        # --- Update priorities ---
        if use_prioritized:
            # td_errors = (q1_pred - target_q).detach().cpu().numpy().squeeze()
            td_errors = torch.abs(q1_pred - target_q).mean(dim=1).detach().cpu().numpy()

            replay_buffer.update_priorities(idxs, td_errors)

        # --- Logging ---
        # log reward stats every 80 epsiodes approx
        if self.total_train_steps % 20000 == 0: # every 80 full epsiodes 
            rewards_np = rewards.cpu().numpy()

            # Scalar statistics
            self.writer.add_scalar("reward_stats/mean", rewards_np.mean(), self.total_train_steps)
            self.writer.add_scalar("reward_stats/std", rewards_np.std(), self.total_train_steps)
            self.writer.add_scalar("reward_stats/skew", scipy.stats.skew(rewards_np), self.total_train_steps)
            self.writer.add_scalar("reward_stats/kurtosis", scipy.stats.kurtosis(rewards_np), self.total_train_steps)

            # Histogram (for full distribution)
            self.writer.add_histogram("reward_stats/histogram", rewards, self.total_train_steps, bins=50)
            
            # self.log_param_stats(self.q1, "q1")
            # self.log_param_stats(self.q2, "q2")
            # self.log_param_stats(self.q1_target, "q1_target")
            # self.log_param_stats(self.q2_target, "q2_target")
            # self.log_param_stats(self.actor, "actor")

            
        # log everything else
        if self.total_train_steps % 1000 == 0:

            # log probs
            self.writer.add_scalar("policy/log_prob_mean", log_probs.mean().item(), self.total_train_steps)
            self.writer.add_scalar("policy/log_prob_std", log_probs.std().item(), self.total_train_steps)

            # # mean and stddev (TEMP)
            # for i in range(mean.shape[1]):
            #     self.writer.add_scalar(f"policy/mean_action_{i}", mean[:, i].mean().item(), self.total_train_steps)
            #     self.writer.add_scalar(f"policy/std_action_{i}", std[:, i].mean().item(), self.total_train_steps)
            
            # mean and stddev of u and a:
            u_residual = u - mean  # remove effect of mean shift due to many samples in batch
            u_std_residual = u_residual.std(dim=0)

            for i in range(u.shape[1]):
                # self.writer.add_scalar(f"policy/u_mean_{i}", u[:, i].mean().item(), self.total_train_steps)
                # self.writer.add_scalar(f"policy/u_std_{i}", u[:, i].std().item(), self.total_train_steps)
                # self.writer.add_scalar(f"policy/u_sampling_std_{i}", u_residual[:, i].std().item(), self.total_train_steps)
                self.writer.add_scalar(f"policy/a_mean_{i}", new_actions[:, i].mean().item(), self.total_train_steps)
                self.writer.add_scalar(f"policy/a_std_{i}", new_actions[:, i].std().item(), self.total_train_steps)

            self.log_metrics(
                actor_loss=actor_loss.item(),
                q1_loss=q1_loss.item(),
                q2_loss=q2_loss.item(),
                alpha=self.alpha.item(),
                alpha_loss=alpha_loss.item()
            )

            # log q1 and q2 output value means to check for exploding values
            self.writer.add_scalar("q_values/q1_mean", q1_val.mean().item(), self.total_train_steps)
            self.writer.add_scalar("q_values/q2_mean", q2_val.mean().item(), self.total_train_steps)
            
        self.total_train_steps += 1


    def log_metrics(self, episode_reward=None, episode_length=None, actor_loss=None,
                q1_loss=None, q2_loss=None, alpha=None, alpha_loss=None,
                reward_dist=None, reward_goal=None, reward_safety=None):

        if reward_dist is not None:
            self.writer.add_scalar("reward/dist_error", reward_dist, self.total_episodes)
        if reward_goal is not None:
            self.writer.add_scalar("reward/goal_bonus", reward_goal, self.total_episodes)
        if reward_safety is not None:
            self.writer.add_scalar("reward/safety_reward", reward_safety, self.total_episodes)

        if episode_reward is not None:
            self.writer.add_scalar("reward/episode", episode_reward, self.total_episodes)
        if episode_length is not None:
            self.writer.add_scalar("episode/length", episode_length, self.total_episodes)
        if actor_loss is not None:
            self.writer.add_scalar("loss/actor", actor_loss, self.total_train_steps)
        if q1_loss is not None:
            self.writer.add_scalar("loss/q1", q1_loss, self.total_train_steps)
        if q2_loss is not None:
            self.writer.add_scalar("loss/q2", q2_loss, self.total_train_steps)
        if alpha is not None:
            self.writer.add_scalar("entropy/alpha", alpha, self.total_train_steps)
        if alpha_loss is not None:  
            self.writer.add_scalar("entropy/alpha_loss", alpha_loss, self.total_train_steps)

    def log_param_stats(self, network, name):
        all_params = []

        for _, param in network.named_parameters():
            all_params.append(param.detach().cpu().flatten())

        # Concatenate into a single tensor
        if all_params:
            all_data = torch.cat(all_params).numpy()
            self.writer.add_scalar(f"{name}/mean", all_data.mean(), self.total_train_steps)
            self.writer.add_scalar(f"{name}/std", all_data.std(), self.total_train_steps)
            self.writer.add_scalar(f"{name}/skew", scipy.stats.skew(all_data), self.total_train_steps)
            self.writer.add_scalar(f"{name}/kurtosis", scipy.stats.kurtosis(all_data), self.total_train_steps)

    def save_model(self, dir_path, normalization_stats=None):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dir_path, 'actor.pth'))
        torch.save(self.q1.state_dict(), os.path.join(dir_path, 'q1.pth'))
        torch.save(self.q2.state_dict(), os.path.join(dir_path, 'q2.pth'))
        torch.save(self.q1_target.state_dict(), os.path.join(dir_path, 'q1_target.pth'))
        torch.save(self.q2_target.state_dict(), os.path.join(dir_path, 'q2_target.pth'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(dir_path, 'actor_optimizer.pth'))
        torch.save(self.q1_optimizer.state_dict(), os.path.join(dir_path, 'q1_optimizer.pth'))
        torch.save(self.q2_optimizer.state_dict(), os.path.join(dir_path, 'q2_optimizer.pth'))
        torch.save(self.log_alpha, os.path.join(dir_path, 'log_alpha.pth'))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(dir_path, 'alpha_optimizer.pth'))

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            np.savez_compressed(
                os.path.join(dir_path, 'replay_buffer.npz'),
                state=self.replay_buffer.state,
                action=self.replay_buffer.action,
                reward=self.replay_buffer.reward,
                next_state=self.replay_buffer.next_state,
                done=self.replay_buffer.done,
                ptr=self.replay_buffer.ptr,
                size=self.replay_buffer.size,
                priorities=self.replay_buffer.priorities,
                max_priority=self.replay_buffer.max_priority
            )
        else:
            # Default uniform buffer
            np.savez_compressed(
                os.path.join(dir_path, 'replay_buffer.npz'),
                state=self.replay_buffer.state,
                action=self.replay_buffer.action,
                reward=self.replay_buffer.reward,
                next_state=self.replay_buffer.next_state,
                done=self.replay_buffer.done,
                ptr=self.replay_buffer.ptr,
                size=self.replay_buffer.size
            )


        if normalization_stats is not None:
            mean, var, count = normalization_stats
            np.savez_compressed(
                os.path.join(dir_path, 'normalization_stats.npz'),
                mean=mean,
                var=var,
                count=count,
                is_fixed=(count is None)  # Store whether FixedNormalizeObservation was used
            )





    def load_model(self, dir_path, use_prioritized=False, load_buffer=True): # dont load buffer when reward has changed!!
        # Load model weights
        self.actor.load_state_dict(torch.load(os.path.join(dir_path, 'actor.pth'), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(dir_path, 'q1.pth'), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(dir_path, 'q2.pth'), map_location=self.device))
        self.q1_target.load_state_dict(torch.load(os.path.join(dir_path, 'q1_target.pth'), map_location=self.device))
        self.q2_target.load_state_dict(torch.load(os.path.join(dir_path, 'q2_target.pth'), map_location=self.device))

        # Load actor/critic optimizers
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(dir_path, 'actor_optimizer.pth'), map_location='cpu'))
        self.q1_optimizer.load_state_dict(torch.load(os.path.join(dir_path, 'q1_optimizer.pth'), map_location='cpu'))
        self.q2_optimizer.load_state_dict(torch.load(os.path.join(dir_path, 'q2_optimizer.pth'), map_location='cpu'))

        # Move optimizer state tensors to correct device and dtype
        for opt in [self.actor_optimizer, self.q1_optimizer, self.q2_optimizer]:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(dtype=torch.float32, device=self.device)

        # Load log_alpha with dtype safety
        loaded_alpha = torch.load(os.path.join(dir_path, 'log_alpha.pth'), map_location='cpu')
        self.log_alpha = loaded_alpha.to(dtype=torch.float32, device=self.device).detach().requires_grad_()

        # Recreate and load alpha optimizer
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(dir_path, 'alpha_optimizer.pth'), map_location='cpu'))
        for state in self.alpha_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dtype=torch.float32, device=self.device)

        # Load replay buffer if available
        if load_buffer:
            buffer_path = os.path.join(dir_path, 'replay_buffer.npz')
            if os.path.exists(buffer_path):
                data = np.load(buffer_path)
                if use_prioritized:
                    self.replay_buffer = PrioritizedReplayBuffer(
                        state_dim=data['state'].shape[1],
                        action_dim=data['action'].shape[1],
                        max_size=data['state'].shape[0]
                    )
                    self.replay_buffer.priorities = data['priorities']
                    self.replay_buffer.max_priority = float(data['max_priority'])
                else:
                    self.replay_buffer = UniformReplayBuffer(
                        state_dim=data['state'].shape[1],
                        action_dim=data['action'].shape[1],
                        max_size=data['state'].shape[0]
                    )
                self.replay_buffer.state = data['state']
                self.replay_buffer.action = data['action']
                self.replay_buffer.reward = data['reward']
                self.replay_buffer.next_state = data['next_state']
                self.replay_buffer.done = data['done']
                self.replay_buffer.ptr = int(data['ptr'])
                self.replay_buffer.size = int(data['size'])
        else:
            self.replay_buffer = UniformReplayBuffer(self.state_dim, self.action_dim)


        # Load normalization stats
        norm_path = os.path.join(dir_path, 'normalization_stats.npz')
        if os.path.exists(norm_path):
            stats = np.load(norm_path, allow_pickle=True)
            mean = stats['mean']
            var = stats['var']
            count = stats['count'] if 'count' in stats else None
            is_fixed = stats['is_fixed'].item() if 'is_fixed' in stats else False
            self.normalization_stats = (mean, var, None if is_fixed else count)
        else:
            self.normalization_stats = None





if __name__ == "__main__":
    #  Dummy config
    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Create agent and buffer
    agent = SACAgent(state_dim, action_dim, device="cpu")  # or "cuda"
    buffer = UniformReplayBuffer(state_dim, action_dim)

    # Fill buffer with fake data
    for _ in range(100):
        s = np.random.randn(state_dim)
        a = np.random.uniform(-1, 1, size=action_dim)  # assume actions in [-1, 1]
        r = np.random.randn()
        s_next = np.random.randn(state_dim)
        done = np.random.choice([0, 1])
        buffer.add(s, a, r, s_next, done)

    # Run one training step
    agent.train(buffer, batch_size=batch_size)

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print("Sampled Action:", action)