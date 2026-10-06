"""A small plain residual CNN. Replace this model freely."""
from torch import nn


class Restorer(nn.Module):
    def __init__(self, width=8, depth=3):
        super().__init__()
        if width < 1 or depth < 2:
            raise ValueError("width must be positive and depth must be at least 2")
        layers = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(width, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


def build_model(config):
    return Restorer(width=config["width"], depth=config["depth"])
