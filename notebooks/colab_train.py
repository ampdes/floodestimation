# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Flood Estimation - Training on Google Colab
#
# This notebook trains the U-Net flood segmentation model on Google Colab with GPU acceleration.
#
# **Steps:**
# 1. Mount Google Drive
# 2. Clone repository (or upload data)
# 3. Install dependencies
# 4. Split dataset
# 5. Train model
# 6. Save results to Drive

# %% [markdown]
# ## 1. Check GPU Availability

# %%
# !nvidia-smi

# %% [markdown]
# ## 2. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 3. Clone Repository

# %%
# Clone from GitHub
# !git clone git@github.com:ampdes/flooding.git
# %cd flooding

# %% [markdown]
# ## 4. Install Dependencies

# %%
# !pip install -q -r requirements.txt

# %% [markdown]
# ## 5. Upload or Link Data
#
# **Option A: Upload data to Colab** (for small datasets)
# ```python
# from google.colab import files
# uploaded = files.upload()  # Upload your train.zip
# !unzip train.zip
# ```
#
# **Option B: Use data from Drive** (recommended)
# ```bash
# # Copy data from Drive to Colab workspace
# !cp -r /content/drive/MyDrive/flooding_data/train ./
# !cp -r /content/drive/MyDrive/flooding_data/rawdata ./
# ```

# %%
# Copy data from Google Drive (adjust paths as needed)
# !cp -r /content/drive/MyDrive/flooding_data/train ./
# !cp -r /content/drive/MyDrive/flooding_data/rawdata ./

# %% [markdown]
# ## 6. Split Dataset

# %%
# !python src/data/split_data.py

# %% [markdown]
# ## 7. Verify Split

# %%
import pandas as pd

train_df = pd.read_csv('outputs/train_split.csv')
val_df = pd.read_csv('outputs/val_split.csv')
test_df = pd.read_csv('outputs/test_split.csv')

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"\nTotal: {len(train_df) + len(val_df) + len(test_df)} samples")

# %% [markdown]
# ## 8. Test Pretrained Model (Before Training)
#
# Let's test the model with pretrained ImageNet weights to see how it performs before training on flood data.

# %%
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.models.unet import create_model
from torchvision import transforms

# Create model with pretrained ImageNet weights
model = create_model(
    name='unet',
    encoder='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# %% [markdown]
# ### Run Inference on Sample Image

# %%
# Load a sample image
image_path = 'data/train/image/0.jpg'
mask_path = 'data/train/mask/0.png'

# Load and preprocess image
image = Image.open(image_path).convert('RGB')
mask_gt = Image.open(mask_path).convert('L')

# Preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform image
image_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask_gt, cmap='gray')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

axes[2].imshow(prediction, cmap='gray')
axes[2].set_title('Pretrained Model Output\n(Before Training)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("\nNote: The pretrained model has only ImageNet weights (not trained on flood data).")
print("The output will likely be poor - this is expected!")
print("After training on our flood dataset, accuracy will improve significantly.")

# %% [markdown]
# ## 9. Train Model

# %%
# Train with default config
# !python src/training/train.py --config config/train_config.yaml

# %% [markdown]
# ## 10. Monitor Training (Optional)
#
# If you want to visualize training in real-time, you can use TensorBoard:

# %%
# Load TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir outputs/logs

# %% [markdown]
# ## 11. View Training History

# %%
import json
import matplotlib.pyplot as plt

# Load training history
with open('outputs/logs/training_history.json', 'r') as f:
    history = json.load(f)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Metrics
metrics = history['val_metrics']
epochs = range(1, len(metrics) + 1)
for metric_name in ['iou', 'dice', 'f1_score']:
    values = [m[metric_name] for m in metrics]
    axes[1].plot(epochs, values, label=metric_name, marker='o')

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Test Prediction

# %%
# Make prediction on a test image
# !python src/inference/predict.py \
#     --checkpoint outputs/models/best_model.pth \
#     --image data/train/image/0.jpg \
#     --output_dir outputs/predictions

# %% [markdown]
# ## 13. Visualize Results

# %%
from PIL import Image
import matplotlib.pyplot as plt

# Load and display
image = Image.open('data/train/image/0.jpg')
mask_true = Image.open('data/train/mask/0.png')
mask_pred = Image.open('outputs/predictions/0_pred.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask_true, cmap='gray')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

axes[2].imshow(mask_pred, cmap='gray')
axes[2].set_title('Prediction')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Save Results to Google Drive

# %%
# Copy results back to Drive for persistence
# !mkdir -p /content/drive/MyDrive/flooding_results
# !cp -r outputs/models /content/drive/MyDrive/flooding_results/
# !cp -r outputs/logs /content/drive/MyDrive/flooding_results/
# !cp -r outputs/predictions /content/drive/MyDrive/flooding_results/

print("Results saved to Google Drive!")

# %% [markdown]
# ## 15. Download Best Model (Optional)

# %%
from google.colab import files

# Download the best model
files.download('outputs/models/best_model.pth')

# %% [markdown]
# ## Next Steps
#
# After training:
# 1. **Evaluate on test set**: Run predictions on all test images
# 2. **GIS Analysis**: Calculate flood extent, flow direction, depth
# 3. **Export Results**: Generate GeoTIFF, Shapefile, GeoJSON outputs
#
# See the main README.md for detailed instructions on GIS analysis.
