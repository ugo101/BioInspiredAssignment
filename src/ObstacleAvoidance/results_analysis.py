import numpy as np
import torch
import os
import sys
import time

path = '/home/damenadmin/Projects/BioInspiredAssignment/src'
sys.path.append(path)

path = '/home/damenadmin/Projects/Thesis_v2p/src'
sys.path.append(path)

from ObstacleAvoidance.simulation_environment import ObstacleAvoidanceEnv
from SoftActorCritic.soft_actor_critic import SACAgent
from utils import FixedNormalizeObservation
from ObstacleAvoidance.params import simpar
from Models.params import titoneri_parameters


def batch_validation(
    load_path,
    num_episodes=100,
    distance_bounds=20.0,
    device=None
):
    """
    Load a trained SAC agent and evaluate it over multiple randomized episodes.
    Reports success, crash, and failure-without-crash counts.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- 1. Create environment with matching normalization ---
    print("[INFO] Initializing environment...")
    env = ObstacleAvoidanceEnv(
        simpar,
        titoneri_parameters,
        distance_bounds=distance_bounds
    )

    x_std = 4 * distance_bounds / np.sqrt(12)
    psi_std = 2 * np.pi / np.sqrt(12)
    u_std = 2 * titoneri_parameters.u_max / np.sqrt(12)
    v_std = 2 * 0.2 / np.sqrt(12)
    r_std = 2 * 0.3 / np.sqrt(12)
    lookahead_std = 1.0
    lookahead_stds = [lookahead_std] * 24

    mean = np.zeros(6 + 24, dtype=np.float32)
    var = np.array([
        x_std**2, x_std**2, psi_std**2,
        u_std**2, v_std**2, r_std**2,
        *(np.array(lookahead_stds) ** 2)
    ], dtype=np.float32)

    env = FixedNormalizeObservation(env, mean, var, epsilon=1e-8, clip_range=10.0)

    # --- 2. Load trained SAC agent ---
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_layers_actor=[128, 128, 128],
        hidden_layers_critic=[258, 258, 258]
    )
    agent.load_model(load_path, load_buffer=False)
    print(f"[INFO] Loaded SAC model from {load_path}")

    # --- 3. Run validation episodes ---
    success_count = 0
    crash_count = 0
    fail_no_crash_count = 0

    for episode in range(num_episodes):
        try:
            state, _ = env.reset(
                init_pose=None,
                goal_pose=None,
                objects_list=None,
                safety_area_length=None,
                safety_area_width=None,
                safety_area_offset=None
            )
        except ValueError as e:
            print(f"[WARNING] Skipping episode {episode+1}: reset error {e}")
            continue

        success = False
        crashed = False

        for t in np.arange(0, env.t_total, env.dt):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            if info and "reward_terms" in info:
                terms = info["reward_terms"]

                if terms.get("discrete_reward_goal", 0.0) >= 200.0:
                    success = True
                    break  # Once goal achieved, assume safe for rest of episode

                if terms.get("crash_reward", 0.0) <= -10.0:
                    crashed = True
                    break

            if terminated or truncated:
                break

            state = next_state

        # Record classification
        if success:
            success_count += 1
        elif crashed:
            crash_count += 1
        else:
            fail_no_crash_count += 1

        if (episode + 1) % 50 == 0:
            print(f"[INFO] Completed {episode+1}/{num_episodes} episodes.")

    # --- 4. Report results ---
    print("\n=== Validation Results ===")
    print(f"Total Episodes: {num_episodes}")
    print(f"Successes    : {success_count}")
    print(f"Crashes      : {crash_count}")
    print(f"Failures (no crash, no goal): {fail_no_crash_count}")
    print("===========================")


if __name__ == "__main__":
    # Example usage
    load_path = "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/Models/phase_1"
    batch_validation(load_path=load_path, num_episodes=100)
