import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# below limits control numerical stability/actors behaviour

LOG_STD_MIN = -20 # lower limit on log stddev to prevent very small variance (policy then becomes near determinisitic and avoids division by 0
# std = exp(-20) = 2 * 10^-9 
LOG_STD_MAX = 2 # prevent too large variance making policy chaotic (too random) and unstable)
# std = exp(2) = 7.39
EPS = 1e-6 # prevents taking log of zero in log prob calculation

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 256]):
        super().__init__()

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.net = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        x = self.net(state)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)

        log_prob = normal.log_prob(u) - torch.log(1 - a.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # above xplodes when a = !-1, use more stable version below:
        # log_prob = normal.log_prob(u)
        # squash_correction = 2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))
        # log_prob = log_prob - squash_correction
        # log_prob = log_prob.sum(dim=-1, keepdim=True)



        return a, log_prob, torch.tanh(mean), mean, log_std, u # squash mean for when using deterministic policy, mean and stddev and u used for tensorboard logging



if __name__== "__main__":
    # Example usage
    state_dim = 4 # length of observable space
    action_dim = 4 # rpm, alpha for ps and sb
    hidden_dim = 256 # this can be tuned, reduce
    hidden_dim = 128 # this can be tuned, reduce

    actor = ActorNetwork(state_dim, action_dim, hidden_dim)
    state = torch.randn((1, state_dim))
    action, log_prob, mean = actor(state)

    print("Action:", action)
    print("Log Probability:", log_prob)
    print("Mean:", mean)