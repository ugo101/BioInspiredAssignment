import numpy as np
import matplotlib.pyplot as plt
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
from ObstacleAvoidance.sac_training import evaluate  # Make sure your evaluate() is importable
def run_validation(
    load_path,
    init_pose,
    goal_pose,
    safety_area_length,
    safety_area_width,
    safety_area_offset,
    objects_list,
    distance_bounds=20.0,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create environment (no scenario yet)
    env = ObstacleAvoidanceEnv(
        simpar,
        titoneri_parameters,
        distance_bounds=distance_bounds
    )

    # 2. Normalization (matching training setup)
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

    # 3. Reset with user-specified scenario
    print("[INFO] Resetting environment with provided scenario...")
    try:
        env.reset(
            init_pose=init_pose,
            goal_pose=goal_pose,
            objects_list=objects_list,
            safety_area_length=safety_area_length,
            safety_area_width=safety_area_width,
            safety_area_offset=safety_area_offset
        )
        print("[INFO] Environment successfully reset with scenario.")
    except ValueError as e:
        print(f"[ERROR] Could not reset environment: {e}")
        return

    # 4. Load trained agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_layers_actor= [128, 128, 128],
        hidden_layers_critic=[258, 258, 258]
    )
    agent.load_model(load_path, load_buffer=False)
    print(f"[INFO] Loaded model from {load_path}")

    # 5. Create figures
    fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
    fig_act, axs_act = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig_wind, ax_wind = plt.subplots(2, 1, figsize=(10, 6))
    fig_lookahead, ax_lookahead = plt.subplots(4, 2, figsize=(10, 6))

    plt.ion()

    # 6. Run evaluation
    evaluate(
        agent, env,
        fig_traj=fig_traj, ax_traj=ax_traj,
        fig_act=fig_act, axs_act=axs_act,
        fig_vel=fig_vel, axs_vel=axs_vel,
        fig_wind=fig_wind, ax_wind=ax_wind,
        fig_lookahead=fig_lookahead, ax_lookahead=ax_lookahead,
        episodes=1,
        t_total=simpar.t_total,
        init_pose=init_pose,
        goal_pose=goal_pose,
        safety_area_length=safety_area_length,
        safety_area_width=safety_area_width,
        safety_area_offset=safety_area_offset,
        objects_list=objects_list
    )


    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Example scenario
    load_path = "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/Models/phase_1"
    init_pose = np.array([0.0, 0.0, 0.0])
    goal_pose = np.array([10.0, 10.0, np.pi/2])
    safety_area_length = 12.0
    safety_area_width = 8.0
    safety_area_offset = 0.5
    objects_list = [[2.0, 0.0, 1.0], [5.0, 5.0, 2.0], [10, 5, 1], [6, 10, 1]]  # Populate as needed

    # set all above to None
    init_pose = None
    goal_pose = None
    safety_area_length = None
    safety_area_width = None
    safety_area_offset = None
    objects_list = None

    run_validation(
        load_path,
        init_pose,
        goal_pose,
        safety_area_length,
        safety_area_width,
        safety_area_offset,
        objects_list
    )