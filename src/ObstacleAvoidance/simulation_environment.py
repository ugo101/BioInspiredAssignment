import gymnasium as gym
from gymnasium import spaces, Env
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import os
import sys
path = '/Users/ugomunzi/Documents/Projects/BioInspiredAssignment/src'
sys.path.append(path)

path = '/Users/ugomunzi/Documents/Projects/Thesis_v2/src'
sys.path.append(path)

from Models.maneuvering_3dof_model import ManeuveringModel3DOF
from Models.params import titoneri_parameters, thruster_parameters
from Models.thruster_model import Thruster
from ObstacleAvoidance.safety_area_generator import generate_safety_bounds, sample_object_from_area

class ObstacleAvoidanceEnv(Env):
    """
    Custom Environment for obstacle avoidance simulation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sim_params, vessel_params, distance_bounds=10.0):
        super().__init__()

        self.dt = sim_params.dt
        self.t_total = sim_params.t_total
        self.vessel_params = vessel_params
        self.distance_bounds = distance_bounds # these are now the theoretical max distance bounds (as bounding area will now be randmly changed)
        self.safety_area = None
        self.n_objects = None

        # Action = [rpm_ps, alpha_ps, rpm_sb, alpha_sb]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observable space: [x_err_goal, y_err_goal, psi_err_goal, u, v, r, 24 lookahead points]
        # increase to 8 lookahead points (every 45 deg)
        low_lookahead = [-2 * self.distance_bounds] * 24
        high_lookahead = [2 * self.distance_bounds] * 24

        self.observation_space = spaces.Box(
            low=np.array(
                [-self.distance_bounds * 2, -self.distance_bounds * 2, -np.pi,
                -1.0, -1.0, -1.0,
                *low_lookahead],
                dtype=np.float32
            ),
            high=np.array(
                [self.distance_bounds * 2, self.distance_bounds * 2, np.pi,
                1.0, 1.0, 1.0,
                *high_lookahead],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, init_pose=None, goal_pose=None, objects_list=None, safety_area_length=10.0, safety_area_width=10.0, safety_area_offset=1.0):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()

        self.time = 0.0
        self.reward_total = 0.0
        self.done = False

        margin = 1.0
        goal_limit = self.distance_bounds - margin

        if goal_pose is not None:
            self.goal_pose = np.array(goal_pose, dtype=np.float32)
        else:
            goal_x = self.np_random.uniform(-goal_limit, goal_limit)
            goal_y = self.np_random.uniform(-goal_limit, goal_limit)
            goal_psi = self.np_random.uniform(-np.pi, np.pi)
            self.goal_pose = np.array([goal_x, goal_y, goal_psi])

        init_limit = self.distance_bounds - margin
        if init_pose is None:
            x0 = self.np_random.uniform(-init_limit, init_limit)
            y0 = self.np_random.uniform(-init_limit, init_limit)
            psi0 = self.np_random.uniform(-np.pi, np.pi)

        else:
            x0, y0, psi0 = init_pose

        ned_init = np.array([x0, y0, psi0])
        self.init_pose = ned_init
        surge = 0.0
        sway = 0.0
        yaw_rate = 0.0
        body_states_init = np.array([surge, sway, yaw_rate])

        # general global saftey area
        min_dist_to_dock = 3.0
        # min_dist_to_dock = 0.5

        if safety_area_length is None:
            safety_area_length = np.random.uniform(0.0, 2 * self.distance_bounds)
        if safety_area_width is None:

            safety_area_width = np.random.uniform(0.0, 2 * self.distance_bounds)
        if safety_area_offset is None:
            safety_area_offset = np.random.uniform(0.0, 2) # slight lateral offset (from line connecting start to end pose) to add same slight variation (Avoid always symmetrical around this center line)
            # lateral_offset = 0.0  # no lateral offset for now
        # safety_area_length = 1
        # safety_area_width = 1.0
        # lateral_offset = 2.0
        global_area = generate_safety_bounds(ned_init, self.goal_pose, safety_area_length, safety_area_width, safety_area_offset, objects_list=None, dist_to_docks=min_dist_to_dock)
        
        if objects_list is None:
            objects_list = []
            self.n_objects = np.random.randint(0, 11)  # Randomly sample between 1 and 5 objects
            for _ in range(self.n_objects):
                objects_list.append(sample_object_from_area(
                    global_area,
                    self.init_pose, 
                    self.goal_pose, 
                    margin=2.0, 
                    max_radius=5.0
                ))

        else:
            self.n_objects = len(objects_list)

        self.safety_area = generate_safety_bounds(
            ned_init, self.goal_pose, safety_area_length, safety_area_width, safety_area_offset, objects_list=objects_list, dist_to_docks=min_dist_to_dock
        )

        self.vessel = ManeuveringModel3DOF(self.vessel_params, states_init=body_states_init, ned_position_init=ned_init)

        # start with init rpm alpha = 0
        self.thruster_ps = Thruster(thruster_parameters, 'portside', np.zeros(2))
        self.thruster_sb = Thruster(thruster_parameters, 'starboard', np.zeros(2))

        self.start_pose = ned_init

        return self._get_observation(), {}
            
    def _get_observation(self):
        """
        Compute observation vector for the agent:
        - Relative pose errors in body frame
        - Body velocities
        - Wind and current (constant or zero in this example)
        - 24 lookahead distances (normalized)
        """
        if self.safety_area is None:
            raise ValueError("Safety area not initialized.")

        # Vessel pose
        pose = self.vessel.ned_states.state_vector_positions
        x, y, psi = pose
        velocities = self.vessel.body_states.state_vector

        # Relative goal errors in body frame
        dx = self.goal_pose[0] - x
        dy = self.goal_pose[1] - y
        d_psi = np.arctan2(np.sin(self.goal_pose[2] - psi), np.cos(self.goal_pose[2] - psi))

        dx_body = dx * np.cos(psi) + dy * np.sin(psi)
        dy_body = -dx * np.sin(psi) + dy * np.cos(psi)

        # Lookahead directions (24 sectors, every 15 degrees)
        num_sectors = 24
        angles = np.linspace(0, 2*np.pi, num_sectors, endpoint=False)
        directions_body = np.array([[np.cos(a), np.sin(a)] for a in angles])

        lookahead_distances = []
        epsilon = 0.1
        # Use a worst-case diagonal plus small epsilon
        max_probe_distance = np.sqrt((2 * self.distance_bounds) ** 2 + (2 * self.distance_bounds) ** 2) + epsilon

        for d_body in directions_body:
            # Rotate direction to NED frame
            d_ned_x = d_body[0] * np.cos(psi) - d_body[1] * np.sin(psi)
            d_ned_y = d_body[0] * np.sin(psi) + d_body[1] * np.cos(psi)
            d_ned = np.array([d_ned_x, d_ned_y])

            probe_endpoint = np.array([x, y]) + max_probe_distance * d_ned
            probe_line = LineString([(x, y), probe_endpoint])

            intersection = self.safety_area.boundary.intersection(probe_line)

            if intersection.is_empty:
                dist = max_probe_distance + 1.0
            elif isinstance(intersection, Point):
                dist = intersection.distance(Point(x, y))
            else:
                distances = [pt.distance(Point(x, y)) for pt in intersection.geoms]
                dist = np.min(distances)

            # Normalize
            normalization_denominator = np.sqrt((2 * self.distance_bounds)**2 + (2 * self.distance_bounds)**2)
            normalized = dist / normalization_denominator
            lookahead_distances.append(normalized)

        # Mark distances negative if vessel is outside safety area
        if self.check_crash_condition():
            lookahead_distances = [-abs(d) for d in lookahead_distances]

        self.last_lookahead_distances = lookahead_distances

        # Assemble final observation
        obs = np.array([
            dx_body, dy_body, d_psi,
            *velocities,
            *lookahead_distances
        ], dtype=np.float32)

        return obs
    
    def step(self, action):
        rpm_ps, alpha_ps, rpm_sb, alpha_sb = self.unnormalize_action(action)

        tau_ps = self.thruster_ps.simulate_timestep(self.dt, rpm_ps, alpha_ps)
        tau_sb = self.thruster_sb.simulate_timestep(self.dt, rpm_sb, alpha_sb)
        tau = tau_ps + tau_sb

        tau_disturbance = np.zeros(3)  # No disturbance in this example

        # CHANGE BELOW WHEN IMPLEMENTING NON ZERO CURRENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        v_current_body = np.zeros(3)

        self.vessel.simulate_timestep(self.dt, tau, tau_disturbance, v_curr_body=v_current_body)

        obs = self._get_observation()

        reward = self._compute_reward()
        self.reward_total += reward
        self.time += self.dt

        pos_reached, heading_reached, velocity_reached = self._is_goal_reached()
        goal_reached = pos_reached and heading_reached and velocity_reached and not self.check_crash_condition()
        
        # terminated = goal_reached # no crash termination (soft contraint)
        # phase 2 out of bounds temrination
        terminated = self.check_crash_condition() #or goal_reached
        truncated = self.time >= self.t_total

        return obs, reward, terminated, truncated, {'reward_terms': self.individual_rewards}

    def _compute_reward(self):
        pose = self.vessel.ned_states.state_vector_positions
        dx = self.goal_pose[0] - pose[0]
        dy = self.goal_pose[1] - pose[1]
        dist_error = np.sqrt(dx ** 2 + dy ** 2)
        max_dist = 2 * self.distance_bounds
        # scale to 0.1 as saftey bonus reward is in magnitude of 0 to 0.1)
        reward_dist = -dist_error / max_dist * 1 # linear penalty # need to pnealise (instead of awardig inverse distance to inctenvise agent to move). otherwise agent will think 'its good enough if i just stay where i start off at as im receiving rewards already'

        heading_error = np.abs(np.arctan2(
            np.sin(self.goal_pose[2] - pose[2]),
            np.cos(self.goal_pose[2] - pose[2])
        ))

        position_reached, heading_reached, vel_reached = self._is_goal_reached()
        goal_reached = position_reached and heading_reached and vel_reached and not self.check_crash_condition()

        # Exponential weight on heading penalty (strong near goal)
        # kicks in at about 1m (a bit less)
        k = 4.0
        weight_heading = np.exp(-k * dist_error)
        reward_heading = -weight_heading * heading_error / np.pi


        # Velocity penalty only near goal (optional)
        # Velocity penalty only near goal (optional)
        vel_norm = np.linalg.norm(self.vessel.body_states.state_vector)
        weight_velocity = np.exp(-k * dist_error)
        reward_vel = -weight_velocity * np.abs(np.linalg.norm(self.vessel.body_states.state_vector)) / 0.664
        
        # positive rward for getting close to goal kciks in at about 1m) (only kicks in when all three activated, so still keep penalties for learning individual dista nd heading reducing is better)
        d = dist_error / max_dist
        h = heading_error / np.pi
        v = vel_norm / 0.664

        k_goal = 3.0
        # for initial phase reduce k to 1 such that positive reward start becoming non zero at about 7m
        # k_goal = 5 # phase 2.0

        composite_error = 4 * d + h + v # weight distance more to have steeper decay w.r.t distance error
        reward_goal = 1.0*np.exp(-k_goal * composite_error) # too large, overshadows the discrete goal bonus (agent prefers to not arrive at dock causing termination as can pickup more conitnuos goal rewards in long run)
        # phase 2 no conitnuos goal reward:
        # reward_goal = 0.0

        if goal_reached:
            reward_discrete_goal = 200.0
        else:
            reward_discrete_goal = 0.0

        # Crash penalty

        
        # reward_crash = -1.0 if self.check_crash_condition() else 0.0 # continuos reward
        # phase 2 discrete out of bounds termination:
        reward_crash = -10.0 if self.check_crash_condition() else 0.0
        # reward_crash = 0.0

        # Distance shaping reward (APG-inspired) # incentivise staying far from objects
        raw_min_distance = np.min(self.last_lookahead_distances)

        if raw_min_distance < 0:
            reward_safety = 0.0
        else:
            k_obstacle = 3.0 # small enough so some reward still given when close to bound (important for docking with dock object!!)
            reward_safety = np.exp(-k_obstacle * (1 - raw_min_distance)) # as always very small max 0.1 needs to be large enough to conuteract negative dist penalty

        # Reward scale: aim for continuos per step reward magnitude of -1 to 1 (except for large discrete termiantion rewards)

        # yaw rate penalty
        # no more crash temrination penalty as the positive reard safety incteivises agent to remain within bounds
        # divide by total number of reward terms to scale between -1 and 1
        reward = reward_goal + reward_safety + reward_dist 
        # reward = reward_dist + reward_safety
        reward /= 3  # scale reward to be between -0.01 and 0.01 
        # reward += reward_discrete_goal
        reward += reward_crash

        # print(f'distance_reward: {reward_dist}, safety_reward: {reward_safety}, goal_reward: {reward_goal}')

        if reward_discrete_goal > 0:
            print(f"Goal reached! Reward: {reward_discrete_goal}")

        self.individual_rewards = {
            'dist_error': reward_dist,
            'goal_bonus': reward_goal,
            'safety_reward': reward_safety,
            'crash_reward': reward_crash,
            'discrete_reward_goal': reward_discrete_goal,
        }

        return reward
    
    def check_crash_condition(self):
        x, y, _ = self.vessel.ned_states.state_vector_positions
        vessel_point = Point(x, y)

        if self.safety_area is None:
            raise ValueError("Safety area not initialized.")

        # Check if vessel is outside safety area
        if not self.safety_area.contains(vessel_point):
            return True

        return False
    
    def _is_goal_reached(self):
        pose = self.vessel.ned_states.state_vector_positions
        d_pos = np.abs(np.linalg.norm(pose[:2] - self.goal_pose[:2]))
        d_yaw = np.abs(np.arctan2(np.sin(pose[2] - self.goal_pose[2]), np.cos(pose[2] - self.goal_pose[2])))
        vel = np.abs(np.linalg.norm(self.vessel.body_states.state_vector))
        return d_pos < 0.4, d_yaw < np.deg2rad(10.0), vel < 0.05


    def unnormalize_action(self, action):
        n_min, a_min = thruster_parameters.n_min, thruster_parameters.alpha_min
        n_max, a_max = thruster_parameters.n_max, thruster_parameters.alpha_max

        rpm_ps = n_min + (action[0] + 1) * (n_max - n_min) / 2
        alpha_ps = a_min + (action[1] + 1) * (a_max - a_min) / 2
        rpm_sb = n_min + (action[2] + 1) * (n_max - n_min) / 2
        alpha_sb = a_min + (action[3] + 1) * (a_max - a_min) / 2

        return rpm_ps, alpha_ps, rpm_sb, alpha_sb
    
    def normalize_action(self, action):
        """
        Normalize physical action space (RPM, alpha) to [-1, 1] SAC space.

        Args:
            action: list or array of [rpm_ps, alpha_ps, rpm_sb, alpha_sb]

        Returns:
            np.ndarray of normalized actions in [-1, 1]
        """
        n_min, a_min = thruster_parameters.n_min, thruster_parameters.alpha_min
        n_max, a_max = thruster_parameters.n_max, thruster_parameters.alpha_max

        rpm_ps_norm = 2 * (action[0] - n_min) / (n_max - n_min) - 1
        alpha_ps_norm = 2 * (action[1] - a_min) / (a_max - a_min) - 1
        rpm_sb_norm = 2 * (action[2] - n_min) / (n_max - n_min) - 1
        alpha_sb_norm = 2 * (action[3] - a_min) / (a_max - a_min) - 1

        return np.array([rpm_ps_norm, alpha_ps_norm, rpm_sb_norm, alpha_sb_norm], dtype=np.float32)