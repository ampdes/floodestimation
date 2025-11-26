"""
Visualization utilities for flood segmentation results.

This module provides functions for visualizing predictions, overlays, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image
from pathlib import Path


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray = None,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (15, 5)
):
    """
    Visualize image, prediction, and optionally ground truth.

    Args:
        image: Original image (HWC)
        prediction: Predicted mask (HW)
        ground_truth: Ground truth mask (HW), optional
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    n_cols = 3 if ground_truth is not None else 2

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (0, 255, 255),
    save_path: str = None
):
    """
    Overlay mask on image.

    Args:
        image: Original image (HWC)
        mask: Binary mask (HW)
        alpha: Transparency of overlay
        color: Color for flooded areas (RGB)
        save_path: Path to save result

    Returns:
        Overlayed image
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    if save_path:
        Image.fromarray(overlay).save(save_path)

    return overlay


def visualize_comparison(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str = None,
    figsize: tuple = (20, 5)
):
    """
    Visualize comparison showing TP, FP, FN.

    Args:
        image: Original image (HWC)
        prediction: Predicted mask (HW), binary
        ground_truth: Ground truth mask (HW), binary
        save_path: Path to save figure
        figsize: Figure size
    """
    # Ensure binary
    if prediction.max() > 1:
        prediction = (prediction > 127).astype(np.uint8)
    if ground_truth.max() > 1:
        ground_truth = (ground_truth > 127).astype(np.uint8)

    # Calculate TP, FP, FN
    tp = np.logical_and(prediction, ground_truth).astype(np.uint8)
    fp = np.logical_and(prediction, np.logical_not(ground_truth)).astype(np.uint8)
    fn = np.logical_and(np.logical_not(prediction), ground_truth).astype(np.uint8)

    # Create comparison image
    comparison = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    comparison[tp > 0] = [0, 255, 0]     # Green: True Positive
    comparison[fp > 0] = [255, 0, 0]     # Red: False Positive
    comparison[fn > 0] = [0, 0, 255]     # Blue: False Negative

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    axes[3].imshow(comparison)
    axes[3].set_title('Comparison')
    axes[3].axis('off')

    # Add legend
    green_patch = mpatches.Patch(color='green', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    axes[3].legend(handles=[green_patch, red_patch, blue_patch], loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_probability_map(
    image: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
    save_path: str = None,
    figsize: tuple = (15, 5)
):
    """
    Visualize probability map with threshold.

    Args:
        image: Original image (HWC)
        prob_map: Probability map (HW), values in [0, 1]
        threshold: Threshold for binary classification
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Probability map
    im = axes[1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Binary prediction
    binary = (prob_map > threshold).astype(np.uint8)
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title(f'Binary (threshold={threshold})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_side_by_side_comparison(
    image_paths: list,
    prediction_paths: list,
    ground_truth_paths: list = None,
    output_path: str = None,
    max_images: int = 5
):
    """
    Create a grid comparison of multiple images.

    Args:
        image_paths: List of image paths
        prediction_paths: List of prediction paths
        ground_truth_paths: List of ground truth paths (optional)
        output_path: Path to save figure
        max_images: Maximum number of images to show
    """
    n_images = min(len(image_paths), max_images)
    n_cols = 3 if ground_truth_paths else 2

    fig, axes = plt.subplots(n_images, n_cols, figsize=(5*n_cols, 5*n_images))

    if n_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_images):
        # Load images
        image = np.array(Image.open(image_paths[i]))
        prediction = np.array(Image.open(prediction_paths[i]))

        # Display
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(prediction, cmap='gray')
        axes[i, 1].set_title(f'Prediction {i+1}')
        axes[i, 1].axis('off')

        if ground_truth_paths:
            ground_truth = np.array(Image.open(ground_truth_paths[i]))
            axes[i, 2].imshow(ground_truth, cmap='gray')
            axes[i, 2].set_title(f'Ground Truth {i+1}')
            axes[i, 2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training history from JSON file.

    Args:
        history_path: Path to training history JSON
        save_path: Path to save figure
    """
    import json

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Metrics plot
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        metrics = history['val_metrics']
        metric_names = list(metrics[0].keys())

        for metric_name in metric_names[:4]:  # Plot first 4 metrics
            values = [m[metric_name] for m in metrics]
            axes[1].plot(epochs, values, label=metric_name, marker='o')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    """Test visualization functions."""
    print("Testing visualization utilities...")

    # Create dummy data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    prediction = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    ground_truth = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    prob_map = np.random.rand(512, 512)

    # Test visualizations
    print("1. Testing basic prediction visualization...")
    visualize_prediction(image, prediction, ground_truth)

    print("2. Testing overlay visualization...")
    overlay = visualize_overlay(image, prediction)

    print("3. Testing comparison visualization...")
    visualize_comparison(image, prediction, ground_truth)

    print("4. Testing probability map visualization...")
    visualize_probability_map(image, prob_map)

    print("\nVisualization tests completed!")
