import torch.nn as nn

class ChessPolicyNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024), nn.ReLU(),
            nn.Linear(1024, num_moves)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)