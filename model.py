from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        # params
        self.input_size = input_size # 14?

        # Layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze()

