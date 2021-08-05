import torch
import torch.nn as nn


class CCC_loss(nn.Module):
    """Concordance Correlation Coefficient"""

    def __init__(self):
        super(CCC_loss, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, outputs, labels):
        mean_labels = self.mean(labels, axis=0)
        mean_outputs = self.mean(outputs, axis=0)
        var_labels = self.var(labels, axis=0)
        var_outputs = self.var(outputs, axis=0)
        cor = self.mean((outputs - mean_outputs) *
                        (labels - mean_labels), axis=0)
        r = 2*cor / (var_labels + var_outputs + (mean_labels-mean_outputs)**2)
        ccc = sum(r)/2
        return 1 - ccc
