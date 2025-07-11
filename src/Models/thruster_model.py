import numpy as np
import matplotlib.pyplot as plt

class Thruster:

    def __init__(self, thruster_parameters, side="portside", inits_states=[0.0, 0.0]):
        self.params = thruster_parameters
        if side == "portside":
            self.ly = self.params.ly_ps
        elif side == "starboard":
            self.ly = self.params.ly_sb

        self.states = np.array(inits_states)  # [rpm, rad] thruster states [rpm, alpha]

    @property
    def rpm(self):
        return self.states[0]
    
    @property
    def alpha(self):
        return self.states[1]

    def thruster_dynamics(self, n_desired, alpha_desired, dt):
        # Thruster dynamics, input: desired rpm and desired angle
        # Output: actual rpm and actual angle
        # limit the rate of change of rpm and angle using dalpha_max and dn_max (ramp function)
        n = self.states[0]
        alpha = self.states[1]

        dn = np.clip(n_desired - n, -self.params.dn_max * dt, self.params.dn_max * dt)
        dalpha = np.clip(alpha_desired - alpha, -self.params.dalpha_max * dt, self.params.dalpha_max * dt)

        n += dn
        alpha += dalpha

        # clip rpm and angle
        n = np.clip(n, self.params.n_min, self.params.n_max)
        alpha = np.clip(alpha, -self.params.alpha_max, self.params.alpha_max)

        return n, alpha
    
    def thrust_force(self, n, alpha):
        # Thrust force (i.e. control force) in body frame
        # F = [Fx, Fy]
        T = self.params.pt[0] * n + self.params.pt[1] * n ** 2

        if n < 0:
            T = -T

        F_x = T * np.cos(alpha)
        F_y = T * np.sin(alpha)

        # moment due to lateral + longitudinal offset
        F_yaw = self.params.lx * F_y - self.ly * F_x

        F = np.array([F_x, F_y, F_yaw])

        return F
    
    def simulate_timestep(self, dt, n_desired, alpha_desired):
        n, alpha = self.thruster_dynamics(n_desired, alpha_desired, dt)
        self.states = np.array([n, alpha])
        tau = self.thrust_force(n, alpha)

        return tau
    
    def plot_thrust(self):
        # constant rpm
        rpm = self.params.n_max
        alpha = np.linspace(-np.pi, np.pi, 100)
        F = self.thrust_force(rpm, alpha)
        # 3 subplots for thrust force
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(alpha, F[0])
        axs[0].set_title('Thrust force in x')
        axs[0].set_xlabel('Angle (rad)')
        axs[0].set_ylabel('Force (N)')
        axs[0].grid()

        axs[1].plot(alpha, F[1])
        axs[1].set_title('Thrust force in y')
        axs[1].set_xlabel('Angle (rad)')
        axs[1].set_ylabel('Force (N)')
        axs[1].grid()

        axs[2].plot(alpha, F[2])
        axs[2].set_title('Thrust moment')
        axs[2].set_xlabel('Angle (rad)')
        axs[2].set_ylabel('Moment (Nm)')
        axs[2].grid()

        plt.show()

