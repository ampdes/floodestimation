"""
U-Net model implementation using segmentation_models_pytorch.

This module provides a wrapper for creating U-Net models with various encoders.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_model(
    model_name: str = 'unet',
    encoder_name: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 1,
    activation: str = 'sigmoid'
):
    """
    Create a segmentation model using segmentation_models_pytorch.

    Args:
        model_name: Model architecture ('unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus')
        encoder_name: Encoder backbone (e.g., 'resnet34', 'resnet50', 'efficientnet-b3', 'mobilenet_v2')
        encoder_weights: Pretrained weights ('imagenet', None)
        in_channels: Number of input channels (3 for RGB)
        classes: Number of output classes (1 for binary segmentation)
        activation: Output activation function ('sigmoid' for binary, 'softmax' for multi-class)

    Returns:
        PyTorch model
    """
    model_name = model_name.lower()

    # Create model based on architecture
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif model_name == 'unetplusplus' or model_name == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif model_name == 'deeplabv3plus' or model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Supported: unet, unetplusplus, deeplabv3, deeplabv3plus")

    return model


def get_model_info(model):
    """
    Get information about the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }

    return info


class ModelWrapper(nn.Module):
    """
    Wrapper class for segmentation model with utility functions.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict(self, x, threshold: float = 0.5):
        """
        Make binary prediction from input image.

        Args:
            x: Input tensor [B, C, H, W]
            threshold: Threshold for binary classification

        Returns:
            Binary mask [B, 1, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits) if logits.min() < 0 else logits
            preds = (probs > threshold).float()
        return preds

    def predict_proba(self, x):
        """
        Get probability map from input image.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Probability map [B, 1, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits) if logits.min() < 0 else logits
        return probs


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: dict = None,
    device: str = 'cuda'
):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration dictionary
        device: Device to load model to

    Returns:
        Loaded model
    """
    if model_config is None:
        model_config = {
            'model_name': 'unet',
            'encoder_name': 'resnet34',
            'encoder_weights': None,  # Don't load pretrained weights
            'in_channels': 3,
            'classes': 1,
            'activation': 'sigmoid'
        }

    # Create model
    model = create_model(**model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")

    return model


# Alternative encoders and their characteristics
ENCODER_INFO = {
    # ResNet family - good general purpose
    'resnet18': {'params': '11M', 'speed': 'fast', 'accuracy': 'good'},
    'resnet34': {'params': '21M', 'speed': 'fast', 'accuracy': 'good'},
    'resnet50': {'params': '23M', 'speed': 'medium', 'accuracy': 'very good'},
    'resnet101': {'params': '42M', 'speed': 'slow', 'accuracy': 'excellent'},

    # EfficientNet family - efficient and accurate
    'efficientnet-b0': {'params': '4M', 'speed': 'very fast', 'accuracy': 'good'},
    'efficientnet-b3': {'params': '10M', 'speed': 'fast', 'accuracy': 'very good'},
    'efficientnet-b5': {'params': '28M', 'speed': 'medium', 'accuracy': 'excellent'},

    # MobileNet family - lightweight for deployment
    'mobilenet_v2': {'params': '2M', 'speed': 'very fast', 'accuracy': 'fair'},

    # DenseNet family - high accuracy
    'densenet121': {'params': '7M', 'speed': 'medium', 'accuracy': 'very good'},
    'densenet169': {'params': '12M', 'speed': 'slow', 'accuracy': 'excellent'},
}


if __name__ == "__main__":
    """
    Test model creation.
    """
    print("Testing model creation...")

    # Test default U-Net with ResNet34
    model = create_model(
        model_name='unet',
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    # Get model info
    info = get_model_info(model)
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_params_M']:.2f}M")
    print(f"  Trainable parameters: {info['trainable_params_M']:.2f}M")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test with wrapper
    wrapped_model = ModelWrapper(model)
    prediction = wrapped_model.predict(dummy_input, threshold=0.5)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique values in prediction: {torch.unique(prediction)}")

    print("\nModel test completed!")
