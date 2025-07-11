import numpy as np
# original titoneri params used for scaling to DAVe dimensions
class titoneri_parameters:
    rho = 1.225
    m = 16.9
    L = 0.97
    w = 0.3
    x_g = 0.0
    g = 9.81

    # taken from https://pure.tudelft.nl/ws/portalfiles/portal/130835911/PhD_thesis_checked_Zhe_Du.pdf p.40
    I_z = 0.5
    X_du = -1.2
    Y_dv = -49.2
    Y_dr = 0.0
    N_dv = 0.0
    N_dr = -1.8

    lx = -0.42
    ly_sb = 0.08 # define for starbvoard
    ly_ps = -0.08 # define for port

    u_max = 1.0 # approximate

    # drag modelling taken from kasper (includes added coriolis, and all dampign terms)
    px = [1.69860581, -0.33423787, -0.9835434, 39.64262874, 5.7952587, 5.97217394, -84.01724043, 114.8157936, 4.60997461]
    py = [7.36751785, -21.28183953, -15.89028993, 114.08307075, 21.76580654, 137.03281268]

class thruster_parameters:
    n_thrusters = 2

    # thrust polynomial coefficients
    pt = [9.34039644e-06, 3.50944441e-07]

    lx = -0.42
    ly_sb = 0.08 # define for starbvoard
    ly_ps = -0.08 # define for port

    n_min = 0
    n_max = 3000
    dn_max = 1000 # rpm/s
    dalpha_max = np.deg2rad(60) # rad/s
    alpha_min = np.deg2rad(-180) # rad/s
    alpha_max = np.deg2rad(180) # rad

    Tnn = 3.50944441e-07  # used in controllers for thrust calculation (for plotting purposes)
