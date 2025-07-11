import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
import torch
import os
import sys
import time

path = '/Users/ugomunzi/Documents/Projects/BioInspiredAssignment/src'
sys.path.append(path)

path = '/Users/ugomunzi/Documents/Projects/Thesis_v2/src'
sys.path.append(path)

from SoftActorCritic.soft_actor_critic import SACAgent
from SoftActorCritic.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from utils import NormalizeObservation, rescale_action, warmup_observation_normalization, FixedNormalizeObservation
from ObstacleAvoidance.params import simpar
from ObstacleAvoidance.simulation_environment import ObstacleAvoidanceEnv
from Models.params import titoneri_parameters, thruster_parameters

# below training for uniform buffer
def train(env,
          normalizer=None,
          num_episodes=400,
          max_timesteps=int(simpar.t_total / simpar.dt),
          batch_size=256,
          reward_scale=1.0,
          hidden_layers_actor=[256, 256],
          hidden_layers_critic=[256, 256],
          actor_lr=3e-4,
          critic_lr=3e-4,
          alpha_lr=3e-4,
          gamma=0.99,
          n_step=1,
          tau=0.005,
          target_entropy=None,
          use_prioritized_buffer=False,
          warmup_steps=None,
          warmup_from_pid = False,
          device=None,
          save_path=None,
          load_path=None,
          evaluate_throughout=True,
          log_path=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    if target_entropy is None:
        target_entropy = -action_dim

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers_actor=hidden_layers_actor,
        hidden_layers_critic=hidden_layers_critic,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        target_entropy=target_entropy,
        device=device,
        log_path=log_path
    )

    if load_path is not None:
        agent.load_model(load_path, use_prioritized=use_prioritized_buffer, load_buffer=False) # set load buffer to True if want to load preivously saved buffEr (do this if reward doesnt change!!!)
        print(f"[INFO] Loaded model from {load_path}")

        if hasattr(agent, "normalization_stats") and agent.normalization_stats is not None:
            mean, var, count = agent.normalization_stats
            if isinstance(env, NormalizeObservation) and count is None:
                raise ValueError("Loaded normalizer is FixedNormalizeObservation but current env uses NormalizeObservation")
            if isinstance(env, FixedNormalizeObservation) and count is not None:
                raise ValueError("Loaded normalizer is NormalizeObservation but current env uses FixedNormalizeObservation")
            env.load_stats(mean, var, count)
            print(f"[INFO] Loaded normalization stats: mean={mean}, var={var}, count={count}")

        replay_buffer = agent.replay_buffer

        if warmup_steps is not None and warmup_steps > 0:
            warmup_buffer_from_policy(env, agent, replay_buffer, warmup_steps, reward_scale, max_timesteps)

    else:
        print("[INFO] Starting training from scratch")
        if use_prioritized_buffer:
            replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, max_size=int(1e6), n_step=n_step, gamma=gamma)
        else:
            replay_buffer = UniformReplayBuffer(state_dim, action_dim, max_size=int(1e6), n_step=n_step, gamma=gamma)

        if isinstance(env, NormalizeObservation):
            print(f"Normalization stats: mean={env.running_mean}, var={env.running_var}, count={env.count}")
        elif isinstance(env, FixedNormalizeObservation):
            print(f"Using fixed normalization stats: mean={env.mean}, var={env.var}")
        print("[INFO] Warmup complete.")

    if warmup_from_pid:
        print("[INFO] Warming up buffer from PID controller samples.")
        warmup_buffer_from_pid(env, replay_buffer, warmup_steps, reward_scale, max_timesteps)
        print("[INFO] PID warmup complete.")

    reward_buffer = []
    reward_window = 100

    def log_smoothed_reward(reward):
        reward_buffer.append(reward)
        if len(reward_buffer) > reward_window:
            reward_buffer.pop(0)
        return np.mean(reward_buffer)

    # Initialize evaluation plots
    fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
    fig_act, axs_act = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig_wind, ax_wind = plt.subplots(2, 1, figsize=(10, 6))
    fig_lookahead, ax_lookahead = plt.subplots(4, 2, figsize=(10, 6))
    plt.ion()  # Enables interactive plotting


    try:
        for episode in range(num_episodes):
            # put a try here as sometimes, when generating safety area, it fails to generate single area containing both start and end pose
            try:
                state, _ = env.reset(seed=None)
            except ValueError as e:
                print(f"[WARNING] Skipping episode {episode+1} due to environment reset error: {e}")
                continue

            episode_reward = 0
            episode_reward_dist = 0.0
            episode_reward_goal = 0.0
            episode_reward_safety = 0.0

            episode_dist_error_terms = []
            episode_goal_bonus_terms = []
            episode_safety_reward_terms = []

            for t in range(max_timesteps):
                action = agent.select_action(state, deterministic=False)
                next_state, reward, terminated, truncated, reward_signal = env.step(action)
                done = terminated or truncated

                # Log per-timestep rewards to TensorBoard
                if reward_signal and 'reward_terms' in reward_signal:
                    terms = reward_signal['reward_terms']
                    episode_dist_error_terms.append(terms.get('dist_error', 0.0))
                    episode_goal_bonus_terms.append(terms.get('goal_bonus', 0.0))
                    episode_safety_reward_terms.append(terms.get('safety_reward', 0.0))

                if reward_signal and 'reward_terms' in reward_signal:
                    terms = reward_signal['reward_terms']
                    episode_reward_dist += terms.get('dist_error', 0.0)
                    episode_reward_goal += terms.get('goal_bonus', 0.0)
                    episode_reward_safety += terms.get('safety_reward', 0.0)

                scaled_reward = reward / reward_scale
                replay_buffer.add(state, action, scaled_reward, next_state, done)

                state = next_state
                episode_reward += reward

                if len(replay_buffer) >= batch_size:
                    if isinstance(replay_buffer, PrioritizedReplayBuffer):
                        agent.train(replay_buffer, batch_size, use_prioritized=True)
                    else:
                        agent.train(replay_buffer, batch_size)

                if done:
                    break

            agent.total_episodes += 1
            smoothed_reward = log_smoothed_reward(episode_reward)
            agent.log_metrics(
                episode_reward=smoothed_reward,
                episode_length=t + 1,
                reward_dist=episode_reward_dist,
                reward_goal=episode_reward_goal,
                reward_safety=episode_reward_safety
            )

            # --- Log per-episode averaged terms to TensorBoard ---
            if episode_dist_error_terms and agent.total_train_steps % 10000 == 0:
                mean_dist_error = np.mean(episode_dist_error_terms)
                mean_goal_bonus = np.mean(episode_goal_bonus_terms)
                mean_safety_reward = np.mean(episode_safety_reward_terms)

                agent.writer.add_scalar('reward_stats/mean_dist_error', mean_dist_error, agent.total_episodes)
                agent.writer.add_scalar('reward_stats/mean_goal_bonus', mean_goal_bonus, agent.total_episodes)
                agent.writer.add_scalar('reward_stats/mean_safety_reward', mean_safety_reward, agent.total_episodes)


            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

            if (episode + 1) % 20 == 0 and evaluate_throughout:
                if isinstance(env, NormalizeObservation):
                    print(f"Normalization stats: mean={env.running_mean}, var={env.running_var}, count={env.count}")
                elif isinstance(env, FixedNormalizeObservation):
                    print(f"Using fixed normalization stats: mean={env.mean}, var={env.var}")
                evaluate(agent, env,
                         fig_traj=fig_traj, ax_traj=ax_traj,
                         fig_act=fig_act, axs_act=axs_act,
                         fig_vel=fig_vel, axs_vel=axs_vel,
                         fig_wind=fig_wind, ax_wind=ax_wind,
                         fig_lookahead=fig_lookahead, ax_lookahead=ax_lookahead,
                         episodes=1,
                         t_total=simpar.t_total)
                
            

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt caught. Saving model before exiting...")

    finally:
        if save_path is not None:
            agent.replay_buffer = replay_buffer
            normalization_stats = normalizer.get_stats() if normalizer is not None else None
            agent.save_model(save_path, normalization_stats=normalization_stats)

        plt.ioff()
        plt.show()

def warmup_buffer_from_policy(env, agent, replay_buffer, warmup_steps, reward_scale, max_timesteps):
    """
    Fill the replay buffer with `warmup_steps` transitions collected
    by running the given agent's policy in the environment.
    
    Args:
        env: the environment
        agent: a trained SACAgent with a policy to use
        replay_buffer: the buffer to fill
        warmup_steps: total number of transitions to add
        reward_scale: scaling factor applied before adding to buffer
        max_timesteps: max steps per episode
    """
    steps_added = 0
    print(f"[INFO] Warming up buffer with {warmup_steps} steps using given policy.")
    while steps_added < warmup_steps:
        try:
            state, _ = env.reset()
        except ValueError as e:
            print(f"[WARNING] Skipping episode during warmup due to reset error: {e}")
            continue
        for _ in range(max_timesteps):
            action = agent.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # if reward > 0:
            #     print(f"[INFO] Warning: High reward {reward:.2f} at step {steps_added}.")
            done = terminated or truncated

            scaled_reward = reward / reward_scale
            replay_buffer.add(state, action, scaled_reward, next_state, done)

            steps_added += 1
            state = next_state

            if done or steps_added >= warmup_steps:
                break
    print(f"[INFO] Completed buffer warmup with {steps_added} steps.")

import matplotlib.pyplot as plt

def warmup_buffer_from_pid(env, replay_buffer, warmup_steps, reward_scale, max_timesteps, split_percentage=0.0):
    """
    Fill the replay buffer with `warmup_steps` transitions using a mix of DP controller
    and random actions. `split_percentage` determines the fraction of steps
    for DP controller; the rest use purely random sampling.
    """
    from controllers.dynamic_positioning_controller import DPController
    from controllers.thrust_allocation import ThrustAllocationController
    from controllers.params import DPControllerParams

    n_pid_steps = int(warmup_steps * split_percentage)
    n_random_steps = warmup_steps - n_pid_steps

    steps_added = 0
    print(f"[INFO] Warming up buffer with {warmup_steps} steps: {n_pid_steps} DP, {n_random_steps} random actions.")

    # === DP Controller Phase ===
    dp_controller = DPController(env.dt, DPControllerParams())
    thrust_allocator = ThrustAllocationController(thruster_parameters)

    while steps_added < n_pid_steps:
        try:
            state, _ = env.reset()
            goal_pose = env.goal_pose
            desired_waypoint = np.append(goal_pose, 0.0)

            for _ in range(max_timesteps):
                vessel = env.vessel
                wind_model = env.wind_model

                current_pose = vessel.ned_states.state_vector_positions
                current_vel = vessel.body_states.state_vector

                wind_force_body, _, env.v_wind, env.beta_wind = wind_model.wind_force(
                    env.v_wind, env.beta_wind, current_vel, current_pose
                )
                tau_wind_obs_body = wind_force_body

                tau_control_des = dp_controller.compute_control(vessel, desired_waypoint)
                tau_control_des -= tau_wind_obs_body

                alpha_ps, rpm_ps, alpha_sb, rpm_sb = thrust_allocator.allocate_thrust(
                    tau_control_des[0], tau_control_des[1], tau_control_des[2]
                )
                action_physical = [rpm_ps, alpha_ps, rpm_sb, alpha_sb]
                action_norm = env.normalize_action(action_physical)

                tau_ps = env.thruster_ps.simulate_timestep(env.dt, rpm_ps, alpha_ps)
                tau_sb = env.thruster_sb.simulate_timestep(env.dt, rpm_sb, alpha_sb)
                tau_total = tau_ps + tau_sb

                env.vessel.simulate_timestep(env.dt, tau_total, tau_wind_obs_body)

                next_state = env.build_observation_from_raw(
                    vessel.ned_states.state_vector_positions,
                    vessel.body_states.state_vector
                )

                reward = env.compute_reward()
                scaled_reward = reward / reward_scale

                position_error = np.linalg.norm(current_pose[:2] - goal_pose[:2])
                heading_error = np.abs(np.arctan2(
                    np.sin(goal_pose[2] - current_pose[2]),
                    np.cos(goal_pose[2] - current_pose[2])
                ))
                velocity = np.linalg.norm(current_vel[:2])
                done = (position_error < 0.1 and heading_error < np.deg2rad(5) and velocity < 0.05)

                replay_buffer.add(state, action_norm, scaled_reward, next_state, done)
                steps_added += 1

                if done or steps_added >= n_pid_steps:
                    break
                state = next_state

        except Exception as e:
            print(f"[WARNING] Skipping DP rollout due to error: {e}")
            continue

    # === Random Policy Phase ===
    while steps_added < warmup_steps:
        try:
            state, _ = env.reset()
        except ValueError as e:
            print(f"[WARNING] Skipping random phase episode due to reset error: {e}")
            continue

        for _ in range(max_timesteps):
            # Uniformly sample action from environment's action space
            action_random = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action_random)
            done = terminated or truncated

            scaled_reward = reward / reward_scale
            replay_buffer.add(state, action_random, scaled_reward, next_state, done)
            steps_added += 1

            if done or steps_added >= warmup_steps:
                break
            state = next_state

    print(f"[INFO] Completed mixed warmup with {steps_added} total steps.")



def compute_reward_for_pid_sample(env, state_vector):
    # Save original environment state
    old_pose = env.vessel.ned_states.state_vector_positions.copy()
    old_vel = env.vessel.body_states.state_vector.copy()

    # Set env state to PID sample
    env.vessel.ned_states.state_vector_positions[:] = state_vector[:3]
    env.vessel.body_states.state_vector[:] = state_vector[3:6]
    
    # Optionally update wind, current if in obs space
    # env.v_wind = state_vector[6]
    # ...

    # Compute reward as if we had taken a step from this state
    reward = env._compute_reward()

    # Restore original environment state
    env.vessel.ned_states.state_vector_positions[:] = old_pose
    env.vessel.body_states.state_vector[:] = old_vel

    return reward



def evaluate(agent, env, fig_traj, ax_traj, fig_act, axs_act, fig_vel, axs_vel, fig_wind=None, ax_wind=None, fig_lookahead=None, ax_lookahead=None,
             episodes=1, episode_id=None, t_total=None, goal_pose=None):
    for _ in range(episodes):
        # put a try here as sometimes, when generating safety area, it fails to generate single area containing both start and end pose
        try:
            init_pose = np.array([0.0, 0.0, 0.0])
            goal_pose = np.array([10.0, 0.0, 0.0])
            dock_on_starboard = True
            safety_area_length = 10.0
            safety_area_width = 8.0
            lateral_offset = 0.0
            # set everything equal to None or phase 2 (ie randomise)
            init_pose = None
            goal_pose = None
            dock_on_starboard = None
            safety_area_length = None
            safety_area_width = None
            lateral_offset = None
            # CHANGE BELOW WHEN DONE TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            state, _ = env.reset(seed=None)
        except ValueError as e:
            print(f"[WARNING] Skipping evaluation episode due to environment reset error: {e}")
            continue
        lookahead_points_normalised = state[-8:]
        goal_pose = env.goal_pose
        safety_area = env.safety_area
        total_reward = 0

        rpm_ps_list, rpm_sb_list = [], []
        alpha_ps_list, alpha_sb_list = [], []

        x_list, y_list, psi_list = [], [], []
        u_list, v_list, r_list = [], [], []
        lookahead_points_list = []
        # wind velocities/angle for validation
        v_wind_list = []
        beta_wind_list = []
        time_list = []
        time = 0.0

        # append initial states
        pose = env.vessel.ned_states.state_vector_positions
        vel = env.vessel.body_states.state_vector
        rpm_ps_list.append(env.thruster_ps.rpm)
        rpm_sb_list.append(env.thruster_sb.rpm)
        alpha_ps_list.append(env.thruster_ps.alpha)
        alpha_sb_list.append(env.thruster_sb.alpha)
        x_list.append(pose[0])
        y_list.append(pose[1])
        psi_list.append(pose[2])
        u_list.append(vel[0])
        v_list.append(vel[1])
        r_list.append(vel[2])
        time_list.append(time)
        lookahead_points_list.append(lookahead_points_normalised)

        

        for t in np.arange(0, t_total, env.dt):
            action = agent.select_action(state, deterministic=True)
            rpm_ps, alpha_ps, rpm_sb, alpha_sb = env.unnormalize_action(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            time += env.dt

            pose = env.vessel.ned_states.state_vector_positions
            vel = env.vessel.body_states.state_vector

            lookahead_points_normalised = next_state[-8:]
            

            rpm_ps_list.append(rpm_ps)
            rpm_sb_list.append(rpm_sb)
            alpha_ps_list.append(alpha_ps)
            alpha_sb_list.append(alpha_sb)
            x_list.append(pose[0])  
            y_list.append(pose[1])
            psi_list.append(pose[2])
            u_list.append(vel[0])
            v_list.append(vel[1])
            r_list.append(vel[2])
            time_list.append(time)
            lookahead_points_list.append(lookahead_points_normalised)

            if done:
                break

        print(f"[Eval] Total Reward: {total_reward:.2f}")

        # Clear plots
        ax_traj.cla()
        for ax in axs_act: ax.cla()
        for ax in axs_vel: ax.cla()

        # Trajectory
        ax_traj.plot(y_list[0], x_list[0], 'go', label="Start")
        ax_traj.plot(y_list, x_list, label="Trajectory")
        ax_traj.plot(goal_pose[1], goal_pose[0], 'rx', label="Goal")

        # === Start and Goal Pose Arrows ===
        arrow_len = 0.8  # Arrow length
        head_width = 0.3
        head_length = 0.4

        # Start pose
        start_pose = [x_list[0], y_list[0], psi_list[0]]
        start_x, start_y, start_psi = start_pose
        ax_traj.arrow(start_y, start_x,
                    arrow_len * np.sin(start_psi),
                    arrow_len * np.cos(start_psi),
                    head_width=head_width, head_length=head_length,
                    fc='green', ec='green', label='Start Heading')

        # Goal pose
        goal_x, goal_y, goal_psi = goal_pose
        ax_traj.arrow(goal_y, goal_x,
                    arrow_len * np.sin(goal_psi),
                    arrow_len * np.cos(goal_psi),
                    head_width=head_width, head_length=head_length,
                    fc='red', ec='red', label='Goal Heading')
        
        # add dock
        # ax_traj.add_patch(plt.Polygon(dock_corners, closed=True, fill='red', edgecolor='blue', label='Dock'))

        # plot saftey area
        def transform_coords(coords):
            """Swap x and y for NED to plotting (East, North)."""
            coords = np.array(coords)
            return np.column_stack((coords[:,1], coords[:,0]))
        def plot_polygon(polygon, **kwargs):
            # Exterior
            exterior_coords = transform_coords(polygon.exterior.coords)
            ax_traj.add_patch(MplPolygon(exterior_coords, closed=True, **kwargs))
            # Interiors (holes)
            for interior in polygon.interiors:
                interior_coords = transform_coords(interior.coords)
                ax_traj.add_patch(MplPolygon(interior_coords, closed=True, facecolor='white', edgecolor='black', alpha=0.5))

        # --- Plot safety area ---
        if isinstance(safety_area, Polygon):
            plot_polygon(safety_area, fill=True, alpha=0.2, edgecolor='black', facecolor='lightblue')


        ax_traj.set_xlabel("East [m]")
        ax_traj.set_ylabel("North [m]")
        ax_traj.set_title("Top-down View (NED)")
        ax_traj.axis("equal")
        ax_traj.grid(True)
        ax_traj.legend()

        # Forces
        axs_act[0].plot(time_list, rpm_ps_list)
        axs_act[1].plot(time_list, rpm_sb_list)
        axs_act[2].plot(time_list, np.rad2deg(alpha_ps_list))
        axs_act[3].plot(time_list, np.rad2deg(alpha_sb_list))
        axs_act[0].set_ylabel("rpm_ps")
        axs_act[1].set_ylabel("rpm_sb")
        axs_act[2].set_ylabel("alpha_ps [deg]")
        axs_act[3].set_ylabel("alpha_sb [deg]")
        axs_act[3].set_xlabel("Time [s]")

        # Velocities
        axs_vel[0].plot(time_list, u_list)
        axs_vel[1].plot(time_list, v_list)
        axs_vel[2].plot(time_list, r_list)
        axs_vel[0].set_ylabel("u [m/s]")
        axs_vel[1].set_ylabel("v [m/s]")
        axs_vel[2].set_ylabel("r [rad/s]")
        axs_vel[2].set_xlabel("Time [s]")
        fig_vel.suptitle("Body Velocities Over Time")

        # plot 8 lookahead points
        for ax in ax_lookahead.flatten():
            ax.cla()
        if fig_lookahead is not None:
            lookahead_points_array = np.array(lookahead_points_list).T
            for i in range(4):
                ax_lookahead[i, 0].plot(time_list, lookahead_points_array[i*2], label=f"Lookahead {i*2+1}")
                ax_lookahead[i, 1].plot(time_list, lookahead_points_array[i*2+1], label=f"Lookahead {i*2+2}")

            for i in range(4):  
                ax_lookahead[i, 0].set_ylabel(f"Lookahead {i*2+1} [m]")
                ax_lookahead[i, 1].set_ylabel(f"Lookahead {i*2+2} [m]")
                ax_lookahead[i, 0].grid(True)
                ax_lookahead[i, 1].grid(True)
                ax_lookahead[i, 0].legend()
                ax_lookahead[i, 1].legend()
            ax_lookahead[3, 0].set_xlabel("Time [s]")
        else:
            print("[WARNING] fig_lookahead is None, not plotting lookahead points.")

        fig_traj.tight_layout()
        fig_act.tight_layout()

        fig_vel.tight_layout()
        fig_traj.canvas.draw()
        fig_act.canvas.draw()
        fig_vel.canvas.draw()
        if fig_wind is not None:
            fig_wind.canvas.draw()
        if fig_lookahead is not None:
            fig_lookahead.canvas.draw()

        fig_traj.canvas.flush_events()
        fig_act.canvas.flush_events()
        fig_vel.canvas.flush_events()

        plt.pause(0.001)



if __name__ == "__main__":

    # wrap below in NormlizeObservation for normalization, however this leads to issues in evaluation (dont have access the the runnign mean and stddev used by normlizer in training)
    # render_env = gym.make(env_name, render_mode="human")  # for evaluation
    distance_bounds = 20.0 # initial trianing
    v_wind_max = 10.0
    v_current_max = 1.0
    # distance_bounds = 10.0 # slowly increase per trianing phase
    env = ObstacleAvoidanceEnv(simpar, titoneri_parameters, distance_bounds=distance_bounds)

    # NORMLIZE OBSERVATION SPACE
    # below uses fix mean and stddev. for uniform distirubtion, stddev is b-a/sqrt(12) and mean = 0
    # Observable space: [x_err_dock, y_error_dock, psi_error_dock, u, v, r, v_wind, beta_wind, v_current, beta_current, 4 lookahead points]
    # for stddev of errors, just use stddev of positions itself using uniform (b-a/sqrt(12))

    # (a) Errors in body frame
    x_mean = 0.0
    x_std = 4 * distance_bounds / np.sqrt(12)
    y_mean = 0.0
    y_std = 4 * distance_bounds / np.sqrt(12)

    # (b) Angle error
    psi_mean = 0.0
    psi_std = 2 * np.pi / np.sqrt(12)

    # (c) Vessel body velocities
    u_mean = 0.0
    u_std = 2 * titoneri_parameters.u_max / np.sqrt(12)
    v_mean = 0.0
    v_std = 2 * 0.2 / np.sqrt(12)
    r_mean = 0.0
    r_std = 2 * 0.3 / np.sqrt(12)

    # (d) Wind magnitude
    v_wind_mean = v_wind_max / 2
    v_wind_std = v_wind_max / np.sqrt(12)
    beta_wind_mean = 0.0
    beta_wind_std = 2 * np.pi / np.sqrt(12)

    # (e) Current magnitude
    v_current_mean = v_current_max / 2
    v_current_std = v_current_max / np.sqrt(12)
    beta_current_mean = 0.0
    beta_current_std = 2 * np.pi / np.sqrt(12)

    # (f) Lookahead distances
    lookahead_max = 2 * distance_bounds
    # lookahead_mean = lookahead_max / 2
    # lookahead_std = lookahead_max / np.sqrt(12)
    # note: lookahead distances normlised to -1 to 1 in environment class already, so use mean 0 and stddev 1 for no normisation here
    lookahead_mean = 0.0
    lookahead_std = 1.0  # stddev of 1 for lookahead distances, as they are normalized to -1 to 1 in the environment
    lookahead_means = [lookahead_mean] * 24
    lookahead_stds = [lookahead_std] * 24

    # --- Assemble
    mean = np.array([
        x_mean, y_mean, psi_mean,
        u_mean, v_mean, r_mean,
        *lookahead_means
    ], dtype=np.float32)

    var = np.array([
        x_std**2, y_std**2, psi_std**2,
        u_std**2, v_std**2, r_std**2,
        *(np.array(lookahead_stds) ** 2)
    ], dtype=np.float32)

    env = FixedNormalizeObservation(env, mean, var, epsilon=1e-8, clip_range=10.0)

    # env = NormalizeObservation(env, epsilon=1e-8, clip_range=10.0)


    normalizer = env  # keep a reference

    # normalizer = None # use this if dont want normlization, comment out above two lines

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high


    num_episodes=100000000
    max_timesteps = int(simpar.t_total / simpar.dt)
    batch_size = 512 # 256

    hidden_dim_actor = 128 # usually can be smaller, also reduces overfitting
    hidden_dim_critic = 258 # might be slightly too small (on edge), critic loss converges to 0 intially then increases and converges to around 25, reward still ok but might be slightly too small

    hidden_layers_actor=[hidden_dim_actor, hidden_dim_actor, hidden_dim_actor] # 4 layers, 256 dim each
    hidden_layers_critic=[hidden_dim_critic, hidden_dim_critic, hidden_dim_critic] # 4 layers, 512 dim each

    # actor_lr = 1e-3
    # critic_lr = 1e-3
    # alpha_lr = 1e-5 # 1e6 too low

    actor_lr = 1e-3
    critic_lr = 1e-3
    alpha_lr = 1e-5 # 1e6 too low


    gamma=0.99
    n_step = 5 # number of steps to look ahead for n-step Q-learning, improves learning stability and speed, decrease as training progresses
    reward_scale = n_step # scale reward by nstep # affects learning rates, for range, look at reward at each step, not cumulative reward per epsiode!!! # aims to scale reward down to around ~-1 (single order magnitude is okay if sometimes a bit higher)
    tau=0.005 # reduce noise in critic loss
    target_entropy = -1*action_dim # for phase 2 increase exploration to get around dock object
    warmup_from_pid = False # if true, the warmup will be done using a PID controller, otherwise, a random policy
    warmup_steps = 0 # when loading model but using new reward structure, warmup buffer with old policy using new reward instead of loading previous buffer!!
    # note: PER works better without warmup steps, as the agent will learn to prefer these 'useless' transitions
    use_prioritized_buffer=False # True # use prioritized replay buffer, but this leads to very high variance in training
    device='cuda'
    # device = 'cpu' # for runpod.io

    
    # below for macOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    # save_path='src/DPController/Models/StaticSetpointRegulation'
    # note: this one below trained with slightly different reward (heading now applied same as velocity) i.e. only at goal pose
    # NOTE: '/home/damenadmin/Projects/SoftActorCritic/src/DPController/Models/StaticSetpointRegulation_v1_changed_reward' has heading reward active only when close to goal pose, doesnt seem to work well (vessel reaches goal but conituosly rotates quickly)
    save_path = '/Users/ugomunzi/Documents/Projects/BioInspiredAssignment/src/ObstacleAvoidance/Models/phase_2'
    # log_path = '/home/damenadmin/Projects/SoftActorCritic/src/DPController/runs' # for tensorboard logging
    log_path = '/Users/ugomunzi/Documents/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs'
    # log_path = None

    # load_path = '/home/damenadmin/Projects/SoftActorCritic/src/DPController/Models/StaticSetpointRegulation' # if not none, willl load model and start training based on saved model
    load_path = '/Users/ugomunzi/Documents/Projects/BioInspiredAssignment/src/ObstacleAvoidance/Models/phase_1'
    load_path = None
    # # Train the agent
    train(env,
          normalizer=normalizer,
          num_episodes=num_episodes,
          max_timesteps=max_timesteps,
          batch_size=batch_size,
          reward_scale=reward_scale,
          hidden_layers_actor=hidden_layers_actor,
          hidden_layers_critic=hidden_layers_critic,
          actor_lr=actor_lr,
          critic_lr=critic_lr,
          alpha_lr=alpha_lr,
          gamma=gamma,
          n_step=n_step,
          tau=tau,
          target_entropy=target_entropy,
          use_prioritized_buffer=use_prioritized_buffer,
          warmup_steps=warmup_steps,
          warmup_from_pid=warmup_from_pid,
          device=device,
          save_path=save_path,
          load_path=load_path,
          evaluate_throughout=True,
          log_path=log_path
          )
