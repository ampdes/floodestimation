"""
Evaluation metrics for flood segmentation.

This module provides common segmentation metrics: IoU, Dice, Pixel Accuracy, Precision, Recall, F1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def iou_score(pred, target, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.

    Args:
        pred: Predicted tensor [B, 1, H, W] (probabilities or logits)
        target: Target tensor [B, 1, H, W] (binary 0 or 1)
        threshold: Threshold for converting probabilities to binary
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score (scalar)
    """
    # Convert predictions to binary
    pred = (pred > threshold).float()
    target = target.float()

    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)

    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def dice_coefficient(pred, target, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate Dice Coefficient (F1 Score for segmentation).

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion
        smooth: Smoothing factor

    Returns:
        Dice coefficient (scalar)
    """
    pred = (pred > threshold).float()
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def pixel_accuracy(pred, target, threshold: float = 0.5):
    """
    Calculate pixel-wise accuracy.

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion

    Returns:
        Pixel accuracy (scalar)
    """
    pred = (pred > threshold).float()
    target = target.float()

    correct = (pred == target).float().sum()
    total = target.numel()

    return (correct / total).item()


def precision(pred, target, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate precision (positive predictive value).

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion
        smooth: Smoothing factor

    Returns:
        Precision score (scalar)
    """
    pred = (pred > threshold).float()
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()

    prec = (true_positive + smooth) / (predicted_positive + smooth)

    return prec.item()


def recall(pred, target, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate recall (sensitivity, true positive rate).

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion
        smooth: Smoothing factor

    Returns:
        Recall score (scalar)
    """
    pred = (pred > threshold).float()
    target = target.float()

    pred = pred.view(-1)
    target = target.view(-1)

    true_positive = (pred * target).sum()
    actual_positive = target.sum()

    rec = (true_positive + smooth) / (actual_positive + smooth)

    return rec.item()


def f1_score(pred, target, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Calculate F1 score (harmonic mean of precision and recall).

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion
        smooth: Smoothing factor

    Returns:
        F1 score (scalar)
    """
    prec = precision(pred, target, threshold, smooth)
    rec = recall(pred, target, threshold, smooth)

    f1 = 2 * (prec * rec) / (prec + rec + smooth)

    return f1


class MetricsCalculator:
    """
    Class to calculate and accumulate metrics over multiple batches.
    """

    def __init__(self, metric_names: list = None, threshold: float = 0.5):
        """
        Initialize metrics calculator.

        Args:
            metric_names: List of metric names to calculate
            threshold: Threshold for binary conversion
        """
        if metric_names is None:
            metric_names = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall', 'f1_score']

        self.metric_names = metric_names
        self.threshold = threshold

        self.metric_functions = {
            'iou': iou_score,
            'dice': dice_coefficient,
            'pixel_accuracy': pixel_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {name: [] for name in self.metric_names}

    def update(self, pred, target):
        """
        Update metrics with a new batch.

        Args:
            pred: Predicted tensor [B, 1, H, W]
            target: Target tensor [B, 1, H, W]
        """
        for metric_name in self.metric_names:
            if metric_name in self.metric_functions:
                metric_fn = self.metric_functions[metric_name]
                value = metric_fn(pred, target, threshold=self.threshold)
                self.metrics[metric_name].append(value)

    def compute(self):
        """
        Compute average metrics.

        Returns:
            Dictionary of average metrics
        """
        avg_metrics = {}
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[metric_name] = np.mean(values)
            else:
                avg_metrics[metric_name] = 0.0

        return avg_metrics

    def get_summary_string(self):
        """
        Get formatted string summary of metrics.

        Returns:
            Formatted string
        """
        avg_metrics = self.compute()
        summary_parts = []

        for metric_name, value in avg_metrics.items():
            summary_parts.append(f"{metric_name}: {value:.4f}")

        return " | ".join(summary_parts)


def calculate_all_metrics(pred, target, threshold: float = 0.5):
    """
    Calculate all metrics at once.

    Args:
        pred: Predicted tensor [B, 1, H, W]
        target: Target tensor [B, 1, H, W]
        threshold: Threshold for binary conversion

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'iou': iou_score(pred, target, threshold),
        'dice': dice_coefficient(pred, target, threshold),
        'pixel_accuracy': pixel_accuracy(pred, target, threshold),
        'precision': precision(pred, target, threshold),
        'recall': recall(pred, target, threshold),
        'f1_score': f1_score(pred, target, threshold)
    }

    return metrics


if __name__ == "__main__":
    """
    Test metrics calculation.
    """
    print("Testing metrics...")

    # Create dummy predictions and targets
    pred = torch.rand(2, 1, 256, 256)  # Random probabilities
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Binary mask

    # Calculate individual metrics
    print("\nIndividual Metrics:")
    print(f"IoU: {iou_score(pred, target):.4f}")
    print(f"Dice: {dice_coefficient(pred, target):.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy(pred, target):.4f}")
    print(f"Precision: {precision(pred, target):.4f}")
    print(f"Recall: {recall(pred, target):.4f}")
    print(f"F1 Score: {f1_score(pred, target):.4f}")

    # Test MetricsCalculator
    print("\nTesting MetricsCalculator...")
    calculator = MetricsCalculator()

    # Simulate multiple batches
    for i in range(3):
        pred_batch = torch.rand(2, 1, 256, 256)
        target_batch = torch.randint(0, 2, (2, 1, 256, 256)).float()
        calculator.update(pred_batch, target_batch)

    # Get average metrics
    avg_metrics = calculator.compute()
    print("\nAverage Metrics:")
    for name, value in avg_metrics.items():
        print(f"{name}: {value:.4f}")

    # Get summary string
    print("\nSummary:")
    print(calculator.get_summary_string())

    print("\nMetrics test completed!")
