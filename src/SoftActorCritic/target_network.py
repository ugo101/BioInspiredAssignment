# initialise as a clone of the critic networks, at each time step update using soft update

def soft_update(source_net, target_net, tau):
    for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
