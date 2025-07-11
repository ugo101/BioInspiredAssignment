import numpy as np
import matplotlib.pyplot as plt

from maneuvering_3dof_model import ManeuveringModel3DOF
from params import titoneri_parameters, thruster_parameters
from thruster_model import Thruster

# test model set constant rpm and alpha
thruster_ps = Thruster(thruster_parameters, side="portside", inits_states=[0.0, 0.0])
thruster_sb = Thruster(thruster_parameters, side="starboard", inits_states=[0.0, 0.0])
vessel = ManeuveringModel3DOF(titoneri_parameters, states_init=[0.0, 0.0, 0.0], ned_position_init=[0.0, 0.0, 0.0], thruster_params=thruster_parameters)
dt = 0.1

x_list = []
y_list = []
psi_list = []

u_list = []
v_list = []
r_list = []

rpm_ps_list = []
rpm_sb_list = []
alpha_ps_list = []
alpha_sb_list = []

t_list = []
rpm_const = 10000  # constant rpm for both thrusters
alpha_ps_const = 0  # constant angle for portside thruster
alpha_sb_const = 0  # constant angle for starboard thruster
t = 0
for i in range(100):
    # set constant rpm and alpha
    tau_ps = thruster_ps.simulate_timestep(dt=dt, n_desired=rpm_const, alpha_desired=alpha_ps_const)
    tau_sb = thruster_sb.simulate_timestep(dt=dt, n_desired=rpm_const, alpha_desired=alpha_sb_const)
    tau_control = tau_ps + tau_sb
    # simulate timestep
    vessel.simulate_timestep(dt=dt, tau_control=tau_control, tau_disturbance=np.array([0.0, 0.0, 0.0]), v_curr_body=[0.0, 0.0, 0.0])
    t += dt
    t_list.append(t)

    print(f"Step {i+1}: Body States: {vessel.body_states}, NED States: {vessel.ned_states}")

    # ned pose is x, y, psi
    pose = vessel.ned_states.state_vector_positions
    x_list.append(pose[0])
    y_list.append(pose[1])
    psi_list.append(pose[2])

    # body velocities are u, v, r
    velocities = vessel.body_states.state_vector
    u_list.append(velocities[0])
    v_list.append(velocities[1])
    r_list.append(velocities[2])

    # thruster states
    rpm_ps_list.append(thruster_ps.rpm)
    rpm_sb_list.append(thruster_sb.rpm)
    alpha_ps_list.append(thruster_ps.alpha)
    alpha_sb_list.append(thruster_sb.alpha)

# first figure: top down ned view (flip x and y axis, for arrows do sin, cos not cos, sin)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y_list, x_list, marker='o', markersize=2, label='Path')
for i in range(0, len(x_list) - 1, 10):
    plt.arrow(y_list[i], x_list[i], 
              np.sin(psi_list[i]) * 0.05, 
              np.cos(psi_list[i]) * 0.05, 
              head_width=0.03, head_length=0.05, fc='red', ec='red')
plt.scatter(y_list[0], x_list[0], color='green', label='Start')
plt.scatter(y_list[-1], x_list[-1], color='blue', label='End')
plt.xlabel('East (y)')
plt.ylabel('North (x)')
plt.title('Top Down NED View')
plt.grid()
plt.axis('equal')
plt.legend()
# second figure: body velocities
plt.subplot(1, 2, 2)
plt.plot(t_list, u_list, label='u (Surge)')
plt.plot(t_list, v_list, label='v (Sway)')
plt.plot(t_list, r_list, label='r (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Body Velocities')
plt.title('Body Velocities Over Time')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

