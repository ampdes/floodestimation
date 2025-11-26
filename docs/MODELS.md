# Model Architectures for Flood Segmentation

This document provides a comprehensive overview of different model architectures suitable for flood segmentation, their trade-offs, and implementation guidelines.

## Current Implementation: U-Net

### Architecture Overview
U-Net is the current default architecture, specifically designed for semantic segmentation tasks. It features:
- Encoder-decoder structure with skip connections
- Pretrained encoder for transfer learning
- Excellent performance on small datasets

### Default Configuration
```python
model = create_model(
    model_name='unet',
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid'
)
```

### Performance Characteristics
- **Parameters**: ~21M (ResNet34 encoder)
- **Training Time**: ~2-4 hours on T4 GPU (100 epochs)
- **Inference Speed**: ~50-100 FPS on GPU
- **Memory Usage**: ~4-6 GB GPU RAM (batch size 8, 512x512)

---

## Alternative Encoder Backbones

### 1. ResNet Family

#### ResNet18
- **Parameters**: 11M
- **Speed**: Very fast
- **Accuracy**: Good
- **Use Case**: When speed is critical and accuracy requirements are moderate

```yaml
# config/train_config.yaml
model:
  encoder: resnet18
```

#### ResNet34 (Default)
- **Parameters**: 21M
- **Speed**: Fast
- **Accuracy**: Very good
- **Use Case**: Best balance of speed and accuracy for most applications

#### ResNet50
- **Parameters**: 23M
- **Speed**: Medium
- **Accuracy**: Very good
- **Use Case**: When higher capacity is needed without much speed sacrifice

#### ResNet101
- **Parameters**: 42M
- **Speed**: Slow
- **Accuracy**: Excellent
- **Use Case**: Maximum accuracy, when computational resources are available

**Recommendation**: ResNet34 for POC, ResNet50 for production if accuracy needs improvement.

---

### 2. EfficientNet Family

#### EfficientNet-B0
- **Parameters**: 4M
- **Speed**: Very fast
- **Accuracy**: Good
- **Use Case**: Deployment on edge devices, mobile applications

```yaml
# config/train_config.yaml
model:
  encoder: efficientnet-b0
```

#### EfficientNet-B3
- **Parameters**: 10M
- **Speed**: Fast
- **Accuracy**: Very good to excellent
- **Use Case**: Best efficiency/accuracy trade-off
- **Note**: Better than ResNet34 with fewer parameters

#### EfficientNet-B5
- **Parameters**: 28M
- **Speed**: Medium
- **Accuracy**: Excellent
- **Use Case**: Production systems where accuracy is paramount

**Recommendation**: EfficientNet-B3 is excellent choice for production - better accuracy than ResNet34 with half the parameters.

---

### 3. MobileNet Family

#### MobileNetV2
- **Parameters**: 2M
- **Speed**: Very fast
- **Accuracy**: Fair to good
- **Use Case**: Real-time applications, deployment constraints

```yaml
# config/train_config.yaml
model:
  encoder: mobilenet_v2
```

**Pros**:
- Extremely lightweight
- Fast inference on CPU
- Suitable for mobile/edge deployment

**Cons**:
- Lower accuracy than ResNet/EfficientNet
- May struggle with complex flood boundaries

---

### 4. DenseNet Family

#### DenseNet121
- **Parameters**: 7M
- **Speed**: Medium
- **Accuracy**: Very good
- **Use Case**: When feature reuse is important

#### DenseNet169
- **Parameters**: 12M
- **Speed**: Slow
- **Accuracy**: Excellent
- **Use Case**: Maximum accuracy with moderate parameters

**Note**: DenseNet can be memory-intensive during training despite fewer parameters.

---

## Alternative Architectures

### 1. U-Net++ (UnetPlusPlus)

Enhanced version of U-Net with nested skip connections.

```python
model = create_model(
    model_name='unetplusplus',
    encoder_name='resnet34',
    encoder_weights='imagenet'
)
```

**Pros**:
- Better feature fusion than U-Net
- Improved boundary detection
- Often higher IoU/Dice scores

**Cons**:
- More parameters than U-Net
- Slower training and inference
- Marginal improvement may not justify complexity

**Recommendation**: Try if U-Net accuracy is insufficient and speed is not critical.

---

### 2. DeepLabV3+

State-of-the-art segmentation with atrous spatial pyramid pooling (ASPP).

```python
model = create_model(
    model_name='deeplabv3plus',
    encoder_name='resnet50',
    encoder_weights='imagenet'
)
```

**Pros**:
- Excellent multi-scale feature extraction
- Superior accuracy on complex scenes
- Good at capturing both large and small flooded regions

**Cons**:
- Slower than U-Net
- Higher memory requirements
- More difficult to train

**Use Case**: When accuracy is critical and computational resources are available.

**Expected Performance**:
- IoU: +2-5% over U-Net
- Training time: +50% over U-Net
- Inference time: +30% over U-Net

---

### 3. DeepLabV3

Simpler version of DeepLabV3+ without the decoder refinement.

```python
model = create_model(
    model_name='deeplabv3',
    encoder_name='resnet50',
    encoder_weights='imagenet'
)
```

**Pros**:
- Faster than DeepLabV3+
- Still benefits from ASPP module
- Good multi-scale understanding

**Cons**:
- Lower accuracy than DeepLabV3+
- Still slower than U-Net

---

### 4. YOLOv8-Seg (Not Currently Implemented)

YOLO adapted for instance segmentation.

**Pros**:
- Very fast inference (~50 FPS on GPU for 512x512)
- Single-stage architecture
- Good for real-time applications

**Cons**:
- Primarily designed for instance segmentation, not semantic
- Less accurate boundaries than U-Net
- Requires different training approach

**Implementation Note**: Would require ultralytics library and different training pipeline.

```python
# Future implementation
from ultralytics import YOLO
model = YOLO('yolov8x-seg.pt')
```

**Use Case**: Real-time video flood monitoring, drone footage analysis.

---

## Encoder Comparison Table

| Encoder | Params | Speed | Accuracy | Memory | Best For |
|---------|--------|-------|----------|--------|----------|
| ResNet18 | 11M | ⚡⚡⚡ | ⭐⭐⭐ | Low | Fast prototyping |
| **ResNet34** | **21M** | **⚡⚡⚡** | **⭐⭐⭐⭐** | **Medium** | **Default choice** |
| ResNet50 | 23M | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Higher accuracy |
| ResNet101 | 42M | ⚡ | ⭐⭐⭐⭐⭐ | High | Maximum accuracy |
| EfficientNet-B0 | 4M | ⚡⚡⚡ | ⭐⭐⭐ | Low | Edge deployment |
| **EfficientNet-B3** | **10M** | **⚡⚡⚡** | **⭐⭐⭐⭐⭐** | **Low** | **Production** |
| EfficientNet-B5 | 28M | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | High accuracy |
| MobileNetV2 | 2M | ⚡⚡⚡⚡ | ⭐⭐ | Very Low | Mobile/Real-time |
| DenseNet121 | 7M | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Feature reuse |
| DenseNet169 | 12M | ⚡ | ⭐⭐⭐⭐⭐ | High | Dense features |

---

## Architecture Comparison

| Architecture | Complexity | Accuracy | Speed | Best For |
|-------------|-----------|----------|-------|----------|
| **U-Net** | **Medium** | **⭐⭐⭐⭐** | **⚡⚡⚡** | **Default, balanced** |
| U-Net++ | High | ⭐⭐⭐⭐⭐ | ⚡⚡ | Better boundaries |
| DeepLabV3 | High | ⭐⭐⭐⭐ | ⚡⚡ | Multi-scale |
| DeepLabV3+ | Very High | ⭐⭐⭐⭐⭐ | ⚡ | Maximum accuracy |
| YOLOv8-Seg | Medium | ⭐⭐⭐ | ⚡⚡⚡⚡ | Real-time |

---

## Recommendations by Use Case

### 1. Proof of Concept (Current)
**Recommended**: U-Net + ResNet34
- Fast iteration
- Good baseline performance
- Easy to train

### 2. Production Deployment
**Recommended**: U-Net + EfficientNet-B3
- Best efficiency/accuracy trade-off
- Lower memory footprint
- Faster inference than ResNet34

```yaml
# config/train_config.yaml
model:
  name: unet
  encoder: efficientnet-b3
  encoder_weights: imagenet
```

### 3. Maximum Accuracy
**Recommended**: DeepLabV3+ + ResNet101 or EfficientNet-B5
- State-of-the-art accuracy
- Complex scene understanding
- Requires more computational resources

```yaml
# config/train_config.yaml
model:
  name: deeplabv3plus
  encoder: resnet101
  encoder_weights: imagenet

training:
  batch_size: 4  # Reduce due to memory
  epochs: 150    # May need more epochs
```

### 4. Real-Time Applications
**Recommended**: U-Net + MobileNetV2 or YOLOv8-Seg
- Fast inference (>30 FPS)
- Suitable for video streams
- CPU-friendly

### 5. Mobile/Edge Deployment
**Recommended**: U-Net + EfficientNet-B0
- Lightweight
- Can run on mobile devices
- Acceptable accuracy

---

## Ensemble Methods

For maximum accuracy, combine multiple models:

```python
# Example ensemble
models = [
    create_model('unet', 'resnet34', 'imagenet'),
    create_model('unet', 'efficientnet-b3', 'imagenet'),
    create_model('deeplabv3plus', 'resnet50', 'imagenet')
]

# Average predictions
predictions = [model(image) for model in models]
final_prediction = torch.mean(torch.stack(predictions), dim=0)
```

**Expected Improvement**: +2-3% IoU over single best model

**Cost**: 3x inference time

---

## Training Tips by Architecture

### U-Net
```yaml
training:
  batch_size: 8
  learning_rate: 0.0001
  loss:
    type: hybrid
    bce_weight: 0.5
    dice_weight: 0.5
```

### DeepLabV3+
```yaml
training:
  batch_size: 4           # Lower due to memory
  learning_rate: 0.00005  # Lower initial LR
  loss:
    type: focal           # Better for hard examples
    alpha: 0.25
    gamma: 2.0
```

### Lightweight Models (MobileNet, EfficientNet-B0)
```yaml
training:
  batch_size: 16          # Can increase batch size
  learning_rate: 0.0002   # Can use higher LR
  epochs: 150             # May need more epochs
```

---

## Switching Architectures

To switch architectures, simply modify `config/train_config.yaml`:

```yaml
# Example: Switch to EfficientNet-B3
model:
  name: unet
  encoder: efficientnet-b3
  encoder_weights: imagenet
```

Then retrain:
```bash
python src/training/train.py --config config/train_config.yaml
```

---

## Performance Benchmarks (Expected)

Based on similar flood segmentation tasks:

| Model | IoU | Dice | Training Time | Inference (FPS) |
|-------|-----|------|--------------|-----------------|
| U-Net + ResNet34 | 0.65 | 0.75 | 2-3h | 80 |
| U-Net + EfficientNet-B3 | 0.68 | 0.78 | 2-3h | 70 |
| U-Net + ResNet50 | 0.67 | 0.77 | 3-4h | 60 |
| DeepLabV3+ + ResNet50 | 0.70 | 0.80 | 4-5h | 40 |
| U-Net + MobileNetV2 | 0.60 | 0.70 | 1.5-2h | 120 |

*Benchmarks on T4 GPU, 512x512 images, 100 epochs*

---

## Future Architectures to Consider

### 1. Transformer-Based Models
- **SegFormer**: Vision transformer for segmentation
- **Swin-UNet**: Swin transformer with U-Net architecture
- **Pros**: State-of-the-art accuracy
- **Cons**: Require more data, slower training

### 2. Attention-Based U-Net
- Add attention gates to U-Net
- Better focus on relevant features
- Moderate complexity increase

### 3. HRNet
- Maintains high-resolution representations
- Excellent for detailed boundaries
- Higher computational cost

---

## Conclusion

**For this project (203 samples):**
1. **Start with**: U-Net + ResNet34 (current default)
2. **Upgrade to**: U-Net + EfficientNet-B3 (if accuracy insufficient)
3. **Maximum accuracy**: DeepLabV3+ + ResNet50 (if resources available)
4. **Deployment**: U-Net + EfficientNet-B0 (for edge devices)

The modular design allows easy experimentation with different architectures by simply changing configuration files.
