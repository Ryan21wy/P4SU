import torch
import torch.nn as nn


class SparseLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SparseLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        if self.reduction == 'mean':
            return torch.mean(torch.norm(x, p=2, dim=0))
        elif self.reduction == 'sum':
            return torch.sum(torch.norm(x, p=2, dim=0))
        else:
            return torch.norm(x, p=2, dim=0)


class SADLoss(nn.Module):
    def __init__(self, num_bands=156):
        super(SADLoss, self).__init__()
        self.num_bands = num_bands

    def forward(self, input, target):
        """Spectral Angle Distance Objective
        Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'

        Params:
            input -> Output of the autoencoder corresponding to subsampled input
                    tensor shape: (batch_size, num_bands)
            target -> Subsampled input Hyperspectral image (batch_size, num_bands)

        Returns:
            angle: SAD between input and target
        """

        input_norm = torch.sqrt(torch.sum(input.pow(2), dim=1) + 1e-3)
        target_norm = torch.sqrt(torch.sum(target.pow(2), dim=1) + 1e-3)

        summation = torch.sum(input * target, dim=1)

        cos = summation / (input_norm * target_norm)
        angle = torch.acos(cos)

        return torch.mean(angle)