# Custom PyTorch loss functions for weighted classification and regression tasks.
import torch
from torch import Tensor
from torch.nn.modules import Module


class CustomBCELoss(Module):
    """Weighted binary cross-entropy loss with optional class imbalance correction.

    pos_weight > 1 increases the penalty for missing positive samples,
    compensating for class imbalance (e.g. pos_weight = n_neg / n_pos).
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super(CustomBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        # Normalize weights so they sum to 1
        weights = weights / torch.sum(weights)
        # Clip predictions to avoid log(0)
        yp = torch.clip(output, 1e-7, 1 - 1e-7)
        term_0 = (1 - target) * torch.log(1 - yp + 1e-7)
        term_1 = self.pos_weight * target * torch.log(yp + 1e-7)
        return -torch.sum(weights * (term_0 + term_1))


class CustomMSELoss(Module):
    """Weighted mean squared error loss."""

    def __init__(self) -> None:
        super(CustomMSELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        # Normalize weights and compute weighted squared error
        weights = weights / torch.sum(weights)
        return torch.sum(weights * torch.pow(output - target, 2), axis=0)


class PinballLoss(Module):
    """Pinball (quantile) loss for quantile regression at a given alpha level."""

    def __init__(self, alpha) -> None:
        super(PinballLoss, self).__init__()
        self.alpha = alpha  # Target quantile level (e.g., 0.1 for the 10th percentile)

    def forward(self, yh: Tensor, y: Tensor, weights: Tensor) -> Tensor:
        difference = y - yh
        weighted_difference = difference * weights
        # Penalize under-predictions with alpha, over-predictions with (1 - alpha)
        loss_positive = self.alpha * weighted_difference[difference >= 0]
        loss_negative = (1 - self.alpha) * -weighted_difference[difference < 0]
        loss_sum = torch.sum(loss_positive) + torch.sum(loss_negative)
        return loss_sum


class TitledLoss(Module):
    """Tilted loss (alias for pinball loss) for quantile regression."""

    def __init__(self, quantile) -> None:
        super(TitledLoss, self).__init__()
        self.quantile = quantile

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        # Normalize weights and apply the tilted loss formula
        weights = weights / torch.sum(weights)
        e = (target - output)
        return torch.mean(weights * torch.maximum(self.quantile * e, (self.quantile - 1) * e))
