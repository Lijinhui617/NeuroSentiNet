import torch.nn as nn
import torch

class EmotionWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets, weights):
        loss = self.ce(outputs, targets)
        weighted = loss * weights
        return torch.mean(weighted)
