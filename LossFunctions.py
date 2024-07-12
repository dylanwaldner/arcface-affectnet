import torch
import torch.nn as nn

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, input, target):
        x = input - torch.mean(input)
        y = target - torch.mean(target)
        x_norm = torch.norm(x, p=2)
        y_norm = torch.norm(y, p=2)
        pearson_correlation = torch.sum(x * y) / (x_norm * y_norm)
        
        # Minimizing (1 - correlation) to maximize the correlation
        return 1 - pearson_correlation

class SignSensitiveMSELoss(nn.Module):
    def __init__(self, same_sign_weight=0.5, diff_sign_weight=2.0):
        super(SignSensitiveMSELoss, self).__init__()
        self.same_sign_weight = same_sign_weight
        self.diff_sign_weight = diff_sign_weight

    def forward(self, input, target):
        # Calculate the square of the differences
        sq_error = (input - target) ** 2
        
        # Check if the predicted and target values have the same sign
        same_sign = (input * target) >= 0
        
        # Apply different weights based on whether signs are the same or different
        weights = torch.where(same_sign, self.same_sign_weight, self.diff_sign_weight)
        
        # Weighted mean squared error
        loss = torch.mean(weights * sq_error)
        return loss
