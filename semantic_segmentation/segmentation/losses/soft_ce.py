from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss,label_smoothed_nll_loss1

__all__ = ["SoftCrossEntropyLoss","SoftCrossEntropyLoss1"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
    
class SoftCrossEntropyLoss1(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    - Support of class weights
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor", "weight"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, 
                 ignore_index: Optional[int] = -100, dim: int = 1, weight: Optional[Tensor] = None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self.register_buffer('weight', weight)  # Register as buffer to work with DDP

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        
        if self.weight is not None:
            # Apply class weights to log probabilities
            log_prob = log_prob * self.weight.view(1, -1, 1, 1)
        
        return label_smoothed_nll_loss1(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
            weight=self.weight  # Pass weights to the loss function
        )
