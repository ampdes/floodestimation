# Google Colab Training Setup Guide

## Quick Start (3 Steps)

### Step 1: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. **File → Upload notebook**
3. Select: `notebooks/colab_train.ipynb`

### Step 2: Prepare Your Data

You have two options:

#### Option A: Upload to Google Drive (Recommended)

```bash
# On your local machine, zip your data
cd /mnt/c/ampdes/mystorage/sandbox/floodestimation
zip -r flood_data.zip data/

# Upload flood_data.zip to Google Drive
# Then in Colab, unzip it
```

#### Option B: Direct Upload to Colab (Faster for small data)

In Colab, run:
```python
from google.colab import files
uploaded = files.upload()  # Upload flood_data.zip
!unzip flood_data.zip
```

### Step 3: Run All Cells

In Colab:
- **Runtime → Change runtime type → GPU (T4)**
- **Runtime → Run all**

Training will start automatically!

---

## Detailed Instructions

### Before You Start

1. **Prepare data locally**:
```bash
cd /mnt/c/ampdes/mystorage/sandbox/floodestimation

# Verify data structure
ls data/train/image/*.jpg | wc -l  # Should show 203
ls data/train/mask/*.png | wc -l   # Should show 203
ls data/train/metadata.csv         # Should exist

# Create a zip file (more efficient for upload)
zip -r flood_data.zip data/
```

2. **Check zip size**:
```bash
ls -lh flood_data.zip
# Should be around 75-80 MB
```

### In Google Colab

#### 1. Enable GPU

- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- GPU type: **T4** (free tier)
- Click **Save**

#### 2. Check GPU Availability

Run first cell:
```python
!nvidia-smi
```

You should see:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   XX°C    P8    XX W / XX W |      0MiB / 15360MiB |      0%      Default |
```

#### 3. Mount Google Drive (If using Drive)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the authentication prompt.

#### 4. Clone Repository

```python
!git clone https://github.com/ampdes/floodestimation.git
%cd floodestimation
```

#### 5. Install Dependencies

```python
!pip install -q -r requirements.txt
```

This takes ~2-3 minutes.

#### 6. Upload/Copy Data

**Option A: From Google Drive**
```python
# Unzip from Drive
!unzip /content/drive/MyDrive/flood_data.zip -d .
!ls data/train/image/ | wc -l  # Verify: should show 203
```

**Option B: Direct upload**
```python
from google.colab import files
uploaded = files.upload()  # Select flood_data.zip
!unzip flood_data.zip
```

#### 7. Split Dataset

```python
!python src/data/split_data.py
```

Expected output:
```
Found 203 images
Valid image-mask pairs: 203
Flood ratio statistics:
  Mean: X.XXXX
  Median: X.XXXX
  ...
Training set:   142 images (70.0%)
Validation set:  30 images (15.0%)
Test set:        31 images (15.0%)
```

#### 8. Start Training

```python
!python src/training/train.py --config config/train_config.yaml
```

Training will begin! Expected output:
```
Using device: cuda
Building model...
Model: unet with resnet34 encoder
Total parameters: 21.XX M
...
Epoch 1/100
  Train Loss: 0.XXXX
  Val Loss:   0.XXXX
  Val Metrics:
    iou: 0.XXXX
    dice: 0.XXXX
    ...
```

#### 9. Monitor Training

Training takes **2-4 hours** on T4 GPU (100 epochs with early stopping).

Watch for:
- **Val IoU** increasing (target: > 0.60)
- **Val Dice** increasing (target: > 0.70)
- Early stopping triggers after ~15 epochs without improvement

#### 10. Save Results to Drive

```python
# Copy trained model back to Drive
!mkdir -p /content/drive/MyDrive/flood_results
!cp -r outputs/models /content/drive/MyDrive/flood_results/
!cp -r outputs/logs /content/drive/MyDrive/flood_results/
!cp outputs/*.csv /content/drive/MyDrive/flood_results/

print("✓ Results saved to Google Drive!")
```

---

## Troubleshooting

### Issue: "No GPU available"

**Solution**:
- Runtime → Change runtime type → GPU
- If still shows CPU, try:
  - Runtime → Factory reset runtime
  - Reconnect and try again
- Free tier limits: ~12-15 hours/day

### Issue: "Out of memory"

**Solution**:
Edit `config/train_config.yaml` before training:
```yaml
training:
  batch_size: 4  # Reduce from 8 to 4
```

### Issue: "Disconnected from runtime"

**Solution**:
- Colab disconnects after ~12 hours or 90 minutes idle
- Results are lost unless saved to Drive
- Always save checkpoints to Drive periodically

### Issue: "Upload too slow"

**Solution**:
- Use Google Drive method (upload once, reuse many times)
- Or reduce data size temporarily for testing
- Colab has good bandwidth from Drive

---

## Expected Training Timeline

| Phase | Time | Details |
|-------|------|---------|
| Setup | 5 min | Install deps, upload data |
| Data split | 1 min | Split 203 samples |
| Training | 2-4 hrs | 100 epochs (or early stop) |
| Save results | 2 min | Copy to Drive |
| **Total** | **2.5-4.5 hrs** | End-to-end |

---

## After Training

### 1. Download Best Model

```python
from google.colab import files
files.download('outputs/models/best_model.pth')
```

### 2. View Training History

```python
import json
import matplotlib.pyplot as plt

with open('outputs/logs/training_history.json') as f:
    history = json.load(f)

# Plot metrics
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

# IoU
metrics = history['val_metrics']
iou = [m['iou'] for m in metrics]
axes[1].plot(iou, label='Val IoU', marker='o')
axes[1].set_title('Validation IoU')
axes[1].legend()
axes[1].grid(True)

plt.show()
```

### 3. Test Predictions

```python
!python src/inference/predict.py \
    --checkpoint outputs/models/best_model.pth \
    --image data/train/image/0.jpg \
    --output_dir outputs/predictions
```

### 4. Visualize Results

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('data/train/image/0.jpg')
gt = Image.open('data/train/mask/0.png')
pred = Image.open('outputs/predictions/0_pred.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title('Image')
axes[1].imshow(gt, cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(pred, cmap='gray')
axes[2].set_title('Prediction')
plt.show()
```

---

## Tips for Better Training

### 1. Monitor Progress

Check validation metrics every 10 epochs:
- If IoU not improving after 20 epochs → adjust learning rate
- If loss is NaN → reduce learning rate significantly

### 2. Adjust Hyperparameters

Edit `config/train_config.yaml`:
```yaml
training:
  learning_rate: 0.0001  # Try 0.00005 if unstable
  batch_size: 8          # Reduce to 4 if OOM
  epochs: 150            # Increase if not converging
```

### 3. Use Different Encoder

For better accuracy, try EfficientNet:
```yaml
model:
  encoder: efficientnet-b3  # Instead of resnet34
```

### 4. Keep Session Active

Colab disconnects after 90 minutes idle. To prevent:
- Keep tab open
- Or use: **Tools → Settings → Enable "Omit code cell output when saving"**
- Run a cell periodically

---

## Cost & Limits

### Free Tier (T4 GPU)
- **GPU Time**: ~12-15 hours/day
- **RAM**: 12 GB
- **Disk**: 100 GB (temporary)
- **Restrictions**: May disconnect after 12 hours

### Colab Pro ($10/month)
- **GPU**: A100 or V100 (much faster)
- **Priority access**: Less likely to disconnect
- **Longer sessions**: Up to 24 hours
- **More GPU time**: ~100 hours/month

For this project: **Free tier is sufficient!**

---

## Next Steps After Training

1. **Evaluate on Test Set**
   - Run predictions on all test images
   - Calculate final metrics (IoU, Dice, F1)

2. **GIS Analysis** (Local or Colab)
   - Calculate flood extent
   - Estimate depth (requires DEM)
   - Export to GeoTIFF/Shapefile

3. **Improve Model** (Optional)
   - Try different encoders
   - Adjust augmentation
   - Ensemble multiple models

---

## Support

- **Colab Issues**: Check [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Training Issues**: See README.md
- **Model Issues**: See docs/MODELS.md

Good luck with training! 🚀
