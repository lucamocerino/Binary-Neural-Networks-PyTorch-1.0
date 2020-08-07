import torch.nn as nn
from .binary_layers import * 

__all__ = ['mlp']

class MLP(nn.Module):
    def __init__(self, out_classes = 10):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 512),
                nn.BatchNorm1d(512, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                BNLinearReLU(512, 256),
                nn.BatchNorm1d(256, eps=1e-4, momentum=0.1, affine=False),
                nn.Linear(256, out_classes),
        )

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def norm_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min = 0.01)
        return

    def forward(self, x):
        self.norm_bn()
        x = self.classifier(x)
        return x

def mlp(out_classes=10):
    return MLP(out_classes)
