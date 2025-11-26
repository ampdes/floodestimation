"""
Loss functions for flood segmentation.

This module provides various loss functions: BCE, Dice, Focal, and Hybrid losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Calculate Dice Loss.

        Args:
            pred: Predicted tensor [B, 1, H, W]
            target: Target tensor [B, 1, H, W]

        Returns:
            Dice loss (scalar)
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # Return 1 - dice as loss
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate Focal Loss.

        Args:
            pred: Predicted tensor [B, 1, H, W] (probabilities or logits)
            target: Target tensor [B, 1, H, W]

        Returns:
            Focal loss
        """
        # Ensure pred is probability (apply sigmoid if needed)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Calculate BCE
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Calculate pt
        pt = torch.exp(-bce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining BCE and Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        """
        Initialize Hybrid Loss.

        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss
        """
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        """
        Calculate Hybrid Loss.

        Args:
            pred: Predicted tensor [B, 1, H, W] (logits)
            target: Target tensor [B, 1, H, W]

        Returns:
            Hybrid loss
        """
        # BCE loss expects logits
        bce = self.bce_loss(pred, target)

        # Dice loss expects probabilities
        pred_prob = torch.sigmoid(pred)
        dice = self.dice_loss(pred_prob, target)

        # Combine losses
        return self.bce_weight * bce + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.
    Useful for addressing class imbalance with different penalties for FP and FN.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Initialize Tversky Loss.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Calculate Tversky Loss.

        Args:
            pred: Predicted tensor [B, 1, H, W]
            target: Target tensor [B, 1, H, W]

        Returns:
            Tversky loss
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky


def get_loss_function(loss_config: dict):
    """
    Factory function to create loss function based on configuration.

    Args:
        loss_config: Dictionary with loss configuration
            Example: {
                'type': 'hybrid',
                'bce_weight': 0.5,
                'dice_weight': 0.5
            }

    Returns:
        Loss function (nn.Module)
    """
    loss_type = loss_config.get('type', 'bce').lower()

    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()

    elif loss_type == 'dice':
        smooth = loss_config.get('smooth', 1e-6)
        return DiceLoss(smooth=smooth)

    elif loss_type == 'focal':
        alpha = loss_config.get('alpha', 0.25)
        gamma = loss_config.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'hybrid':
        bce_weight = loss_config.get('bce_weight', 0.5)
        dice_weight = loss_config.get('dice_weight', 0.5)
        smooth = loss_config.get('smooth', 1e-6)
        return HybridLoss(bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)

    elif loss_type == 'tversky':
        alpha = loss_config.get('alpha', 0.5)
        beta = loss_config.get('beta', 0.5)
        smooth = loss_config.get('smooth', 1e-6)
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. "
                         f"Supported: bce, dice, focal, hybrid, tversky")


if __name__ == "__main__":
    """
    Test loss functions.
    """
    print("Testing loss functions...")

    # Create dummy predictions and targets
    pred_logits = torch.randn(2, 1, 256, 256)  # Logits
    pred_probs = torch.sigmoid(pred_logits)    # Probabilities
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    # Test BCE Loss
    print("\n1. BCE Loss:")
    bce_loss = nn.BCEWithLogitsLoss()
    loss_value = bce_loss(pred_logits, target)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Dice Loss
    print("\n2. Dice Loss:")
    dice_loss = DiceLoss()
    loss_value = dice_loss(pred_probs, target)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Focal Loss
    print("\n3. Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(pred_probs, target)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Hybrid Loss
    print("\n4. Hybrid Loss:")
    hybrid_loss = HybridLoss(bce_weight=0.5, dice_weight=0.5)
    loss_value = hybrid_loss(pred_logits, target)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Tversky Loss
    print("\n5. Tversky Loss:")
    tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
    loss_value = tversky_loss(pred_probs, target)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test factory function
    print("\n6. Testing loss factory function:")
    loss_config = {'type': 'hybrid', 'bce_weight': 0.5, 'dice_weight': 0.5}
    loss_fn = get_loss_function(loss_config)
    loss_value = loss_fn(pred_logits, target)
    print(f"   Loss: {loss_value.item():.4f}")

    print("\nLoss functions test completed!")
