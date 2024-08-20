import torch
import torch.nn as nn


class BalancedMSELoss(nn.Module):

    def __init__(self, scale=True):
        super(BalancedMSELoss, self).__init__()

        self.factor = [1, 0.7, 0.6]

        self.mse = nn.MSELoss()
        if scale:
            self.mse = ScaledMSELoss()
            print("Applying ScaledMSELoss")
        else:
            print("Applying MSELoss without scaling")

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))

        l1 = self.mse(pred[actual[:, 1] == 1], y[actual[:, 1] == 1]) * self.factor[0]
        l2 = self.mse(pred[actual[:, 2] == 1], y[actual[:, 2] == 1]) * self.factor[1]
        l3 = self.mse(pred[actual[:, 3] == 1], y[actual[:, 3] == 1]) * self.factor[2]

        return l1 + l2 + l3

class ScaledMSELoss(nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, y):
        mu = torch.minimum(torch.exp(6 * (y-3)) + 1, torch.ones_like(y) * 5) # Reciprocal of the square root of the original dataset distribution.

        return torch.mean(mu * (y-pred) ** 2)

# class ScaledMSELoss(nn.Module):
#
#     def __init__(self):
#         super(ScaledMSELoss, self).__init__()
#
#     def forward(self, pred, y):
#         mu = torch.minimum(torch.exp(6 * (y-3)) + 1, torch.ones_like(y) * 5) # Reciprocal of the square root of the original dataset distribution.
#
#         return torch.mean(mu * (y-pred) ** 2)
