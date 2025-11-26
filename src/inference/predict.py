"""
Prediction/inference script for flood segmentation.

This script handles making predictions on new images using a trained model.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet import load_model_from_checkpoint
from src.data.augmentation import get_validation_augmentation, get_preprocessing


class FloodPredictor:
    """
    Class for making flood predictions on images.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to training config (optional)
            device: Device to run inference on
            threshold: Threshold for binary classification
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold

        # Load config if provided
        self.config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Load model
        self.load_model()

        # Setup preprocessing
        self.setup_preprocessing()

    def load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Get model config
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
        else:
            # Default config
            model_config = {
                'name': 'unet',
                'encoder': 'resnet34',
                'encoder_weights': None,
                'in_channels': 3,
                'classes': 1,
                'activation': 'sigmoid'
            }

        # Create and load model
        from src.models.unet import create_model
        self.model = create_model(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def setup_preprocessing(self):
        """Setup preprocessing pipeline."""
        if self.config and 'data' in self.config:
            image_size = self.config['data']['image_size']
            normalize = self.config['data']['normalize']
            mean = self.config['data'].get('norm_mean')
            std = self.config['data'].get('norm_std')
        else:
            # Default values
            image_size = 512
            normalize = True
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.image_size = image_size
        self.augmentation = get_validation_augmentation(image_size=image_size)
        self.preprocessing = get_preprocessing(normalize=normalize, mean=mean, std=std)

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for inference.

        Args:
            image: Input image as numpy array (HWC)

        Returns:
            Preprocessed tensor (1CHW)
        """
        # Apply augmentation (resize)
        augmented = self.augmentation(image=image)
        image = augmented['image']

        # Apply preprocessing (normalize + to tensor)
        preprocessed = self.preprocessing(image=image, mask=image)
        image_tensor = preprocessed['image']

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict(self, image_path: str, return_prob: bool = False):
        """
        Make prediction on a single image.

        Args:
            image_path: Path to input image
            return_prob: Whether to return probability map or binary mask

        Returns:
            Predicted mask (HW) as numpy array
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        image = np.array(image)

        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # (H, W)

        # Resize back to original size
        prob = cv2.resize(prob, original_size, interpolation=cv2.INTER_LINEAR)

        if return_prob:
            return prob
        else:
            # Apply threshold
            mask = (prob > self.threshold).astype(np.uint8) * 255
            return mask

    def predict_batch(self, image_paths: list, output_dir: str = None):
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save predictions (optional)

        Returns:
            List of predicted masks
        """
        predictions = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for image_path in tqdm(image_paths, desc="Predicting"):
            # Predict
            mask = self.predict(image_path, return_prob=False)
            predictions.append(mask)

            # Save if output directory provided
            if output_dir:
                image_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{image_name}_pred.png")
                Image.fromarray(mask).save(output_path)

        return predictions

    def predict_folder(self, input_dir: str, output_dir: str, pattern: str = "*.jpg"):
        """
        Predict on all images in a folder.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save predictions
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
        """
        # Get all image paths
        input_path = Path(input_dir)
        image_paths = list(input_path.glob(pattern))

        if not image_paths:
            # Try other extensions
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_paths.extend(input_path.glob(ext))

        image_paths = sorted([str(p) for p in image_paths])

        print(f"Found {len(image_paths)} images")

        # Predict
        self.predict_batch(image_paths, output_dir=output_dir)

        print(f"Predictions saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Predict flood segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create predictor
    predictor = FloodPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        threshold=args.threshold
    )

    # Predict
    if args.image:
        # Single image
        print(f"Predicting on single image: {args.image}")
        mask = predictor.predict(args.image)

        # Save prediction
        os.makedirs(args.output_dir, exist_ok=True)
        image_name = Path(args.image).stem
        output_path = os.path.join(args.output_dir, f"{image_name}_pred.png")
        Image.fromarray(mask).save(output_path)
        print(f"Prediction saved to {output_path}")

    elif args.input_dir:
        # Folder of images
        print(f"Predicting on images in: {args.input_dir}")
        predictor.predict_folder(args.input_dir, args.output_dir)

    else:
        print("Please provide either --image or --input_dir")


if __name__ == "__main__":
    main()
