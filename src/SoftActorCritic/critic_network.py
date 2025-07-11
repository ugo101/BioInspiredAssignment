import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 256]):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))  # Final scalar output
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


if __name__== "__main__":
    q1 = QNetwork(state_dim=4, action_dim=4)
    dummy_state = torch.randn(1, 4)
    dummy_action = torch.randn(1, 4)
    q_value = q1(dummy_state, dummy_action)
    print("Q-value:", q_value)  # Should be shape [1, 1]
