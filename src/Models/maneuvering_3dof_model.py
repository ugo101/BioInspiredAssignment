import numpy as np
import matplotlib.pyplot as plt

# below 2 classes are helpers to be able to retrieve model states by name avoiding indexing errors

class States:
        """Class to represent the states of the vessel."""
        def __init__(self, states_init):
            self.u = states_init[0]
            self.v = states_init[1]
            self.r = states_init[2]

        def __repr__(self):
            return f"States(surge={self.u}, sway={self.v}, yaw={self.r})"
        @property
        def state_vector(self):
            return [self.u, self.v, self.r]
        
class NEDStates:
    """Class to represent the NED position, orientation, and velocities."""
    def __init__(self, ned_states_init, ned_vel_init):
        # Position and orientation
        self.x = ned_states_init[0]
        self.y = ned_states_init[1]
        self.psi = ned_states_init[2]  # rad

        # Velocities in the NED frame
        self.x_dot = ned_vel_init[0]
        self.y_dot = ned_vel_init[1]
        self.psi_dot = ned_vel_init[2]

    def __repr__(self):
        return (f"NEDStates(north={self.x}, east={self.y}, yaw={self.psi}, "
                f"north_vel={self.x_dot}, east_vel={self.y_dot}, yaw_rate={self.psi_dot})")
    @property
    def state_vector_positions(self):
        return [self.x, self.y, self.psi]
    @property
    def state_vector_velocities(self):
        return [self.x_dot, self.y_dot, self.psi_dot]


class ManeuveringModel3DOF():
    
    def __init__(self, parameters, states_init=[0.0, 0.0, 0.0], ned_position_init=[0.0, 0.0, 0.0], thruster_params=None):
        self.params = parameters
        self.thruster_params = thruster_params

        self.body_states = States(states_init)  # [m/s, m/s, rad/s] body states [u, v, r]
        init_ned_velocities = self.rotation_body_to_ned(psi=ned_position_init[2]) @ np.array(self.body_states.state_vector)
        self.ned_states = NEDStates(ned_position_init, init_ned_velocities)  # Position and velocity

        self.M = self.get_mass_matrix()

    def set_body_velocities(self, u, v, r):
        self.body_states.u = u
        self.body_states.v = v
        self.body_states.r = r


    def get_mass_matrix(self):
        # page 135 of Fossen
        mass_rigid_body = np.array([
            [self.params.m, 0.0, 0.0],
            [0.0, self.params.m, self.params.m * self.params.x_g],
            [0.0, self.params.m * self.params.x_g, self.params.I_z]
        ])

        added_mass = np.array([
            [-self.params.X_du, 0.0, 0.0],
            [0.0, -self.params.Y_dv, -self.params.Y_dr],
            [0.0, -self.params.Y_dr, -self.params.N_dr]
        ])

        return mass_rigid_body + added_mass

    def get_c_rigid_body(self):
        c_rigid_body = np.array([
            [0.0, 0.0, -self.params.m * (self.params.x_g * self.body_states.r + self.body_states.v)],
            [0.0, 0.0, self.params.m * self.body_states.u],
            [self.params.m * (self.params.x_g * self.body_states.r + self.body_states.v), -self.params.m * self.body_states.u, 0.0]
        ])

        return c_rigid_body
    
    def get_drag_forces(self, u, v, r):

        def fdragY(u, v, p):
            sign_v = np.sign(v)  # Store original sign of v
            v = np.abs(v)  # Work with absolute value

            z = p[0] * v + \
                p[1] * u * v + p[2] * v ** 2 + \
                p[3] * u ** 2 * v + p[4] * u * v ** 2 + p[5] * v ** 3

            return z * sign_v  # Apply odd symmetry


        def fdragX(u, v, p):
            v = np.abs(v)
            z = p[0] * u + p[1] * v + \
                p[2]*u**2+ p[3] * u * v + p[4] * v ** 2 + \
                p[5]*u**3 + p[6]*u ** 2 * v + p[7]*u * v ** 2 + p[8]* v ** 3
            return z
        
        tau_x = -fdragX(u, v, self.params.px)
        tau_y = -fdragY(u, v, self.params.py)
        tau_r = -self.params.L/3 * fdragY(0, r * self.params.L/3, self.params.py) # taken from kasper code

        return np.array([tau_x, tau_y, tau_r])

    
    def simulate_timestep(self, dt, tau_control, tau_disturbance, v_curr_body=[0.0, 0.0, 0.0]): # timestep arg used to validate model with rosbag inputs
         v = np.array(self.body_states.state_vector) - np.array(v_curr_body)
         tau = tau_control + tau_disturbance

         self.M = self.get_mass_matrix()
         self.C_rb = self.get_c_rigid_body()
         tau_drag = self.get_drag_forces(v[0], v[1], v[2])

         dv = np.linalg.inv(self.M) @ (tau + tau_drag - self.C_rb @ v)

         self.body_states.u += dv[0] * dt
         self.body_states.v += dv[1] * dt
         self.body_states.r += dv[2] * dt

         # update ned_states with new body velocities
         self.update_ned_states(dt)

    ## Helpers

    def update_ned_states(self, dt):
        R = self.rotation_body_to_ned()

        # Body velocities
        v_body = np.array(self.body_states.state_vector)
        
        # Calculate NED velocities
        ned_dot = R @ v_body

        # Update NED positions
        self.ned_states.x += ned_dot[0] * dt
        self.ned_states.y += ned_dot[1] * dt
        self.ned_states.psi += ned_dot[2] * dt

        #wrap psi
        self.ned_states.psi = np.arctan2(np.sin(self.ned_states.psi), np.cos(self.ned_states.psi))

        # Store NED velocities
        self.ned_states.x_dot = ned_dot[0]
        self.ned_states.y_dot = ned_dot[1]
        self.ned_states.psi_dot = ned_dot[2]

        # wrap psi_dot
        self.ned_states.psi_dot = np.arctan2(np.sin(self.ned_states.psi_dot), np.cos(self.ned_states.psi_dot))

    def rotation_body_to_ned(self, psi=None):
        if psi is None:
            psi = self.ned_states.psi
        mat = np.array([
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0]
        ])
        return mat
