"""The same image network for DQN and Double DQN."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn


def configure(device):
    torch.set_num_threads(2 if torch.device(device).type == "cuda" else 1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 3), nn.LeakyReLU(0.01), nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.LeakyReLU(0.01), nn.Linear(512, 4))

    def forward(self, images):
        return self.fc(self.conv(images.float() / 255.0))
