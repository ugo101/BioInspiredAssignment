import numpy as np

class simpar:
    
    dt = 0.3
    t_total = 300 # for moment action is fix so only allow enough time for actuator dynamics (action fix as observation state only tau des which is fix along episode)