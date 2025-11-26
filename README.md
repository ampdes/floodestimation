# Flood Estimation from Aerial Imagery

A comprehensive deep learning and GIS pipeline for detecting and analyzing flood extent from aerial/satellite imagery. This project uses U-Net semantic segmentation to identify flooded areas and integrates with GIS tools to calculate flood extent, water flow direction, and depth estimation.

## Features

- **Deep Learning Segmentation**
  - U-Net with pretrained encoders (ResNet34, EfficientNet, MobileNet)
  - Transfer learning for small datasets
  - Comprehensive data augmentation
  - Multiple loss functions (BCE, Dice, Focal, Hybrid)
  - Full training pipeline with checkpointing and early stopping

- **GIS Analysis**
  - Flood extent calculation (area in m² and km²)
  - Water flow direction analysis using DEM
  - Flood depth estimation
  - Export to multiple GIS formats (GeoTIFF, Shapefile, GeoJSON, KML)

- **Visualization**
  - Prediction overlays on original images
  - Comparison views (TP, FP, FN)
  - Probability maps and confidence visualization
  - Training history plots

## Project Structure

```
floodestimation/
├── config/                      # Configuration files
│   ├── train_config.yaml        # Training parameters
│   └── gis_config.yaml          # GIS analysis settings
├── src/                         # Source code
│   ├── data/                    # Data processing
│   │   ├── split_data.py        # Dataset splitting
│   │   ├── augmentation.py      # Data augmentation
│   │   └── dataset.py           # PyTorch datasets
│   ├── models/                  # Model architectures
│   │   └── unet.py              # U-Net implementation
│   ├── training/                # Training infrastructure
│   │   ├── train.py             # Main training script
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── losses.py            # Loss functions
│   ├── inference/               # Prediction and visualization
│   │   ├── predict.py           # Inference script
│   │   └── visualize.py         # Visualization tools
│   └── gis/                     # GIS analysis
│       ├── extent_analysis.py   # Flood area calculation
│       ├── flow_direction.py    # Flow direction analysis
│       ├── depth_estimation.py  # Depth estimation
│       └── export_utils.py      # GIS format exports
├── train/                       # Training data
│   ├── image/                   # RGB images (203 samples)
│   └── mask/                    # Binary masks
├── rawdata/                     # Original satellite imagery
├── outputs/                     # Generated outputs
├── notebooks/                   # Jupyter notebooks
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Google Colab or cloud GPU access (for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd floodestimation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GIS functionality, install additional tools:
```bash
# WhiteboxTools for flow direction analysis
pip install whitebox

# For DEM download (optional)
pip install py3dep elevation
```

## Quick Start

### 1. Prepare Data

Split the dataset into train/val/test sets:
```bash
python src/data/split_data.py
```

This creates:
- `outputs/train_split.csv` (70% - ~142 images)
- `outputs/val_split.csv` (15% - ~30 images)
- `outputs/test_split.csv` (15% - ~31 images)

### 2. Train Model

Train the U-Net model:
```bash
python src/training/train.py --config config/train_config.yaml
```

Training options:
- Modify `config/train_config.yaml` for hyperparameters
- Model checkpoints saved to `outputs/models/`
- Training logs saved to `outputs/logs/`

**For Cloud Training (Google Colab):**
- Upload data to Google Drive
- Use `notebooks/02_model_training.ipynb`
- Free T4 GPU or upgrade to A100

### 3. Make Predictions

Predict on a single image:
```bash
python src/inference/predict.py \
    --checkpoint outputs/models/best_model.pth \
    --image path/to/image.jpg \
    --output_dir outputs/predictions
```

Predict on a folder:
```bash
python src/inference/predict.py \
    --checkpoint outputs/models/best_model.pth \
    --input_dir test_images/ \
    --output_dir outputs/predictions
```

### 4. GIS Analysis

#### Calculate Flood Extent
```bash
python src/gis/extent_analysis.py \
    --mask outputs/predictions/image_pred.png \
    --source_tiff rawdata/2021-09-02_strip_4861038_composite.tif \
    --output_dir outputs/gis
```

Outputs:
- Total flooded area in m² and km²
- Number of connected flood components
- Statistics per component

#### Estimate Flood Depth
```bash
python src/gis/depth_estimation.py \
    --mask outputs/predictions/image_pred.png \
    --dem path/to/dem.tif \
    --output_dir outputs/gis \
    --method boundary
```

Outputs:
- Depth map as GeoTIFF
- Mean, median, max depth statistics
- Depth distribution visualization

#### Analyze Flow Direction
```bash
python src/gis/flow_direction.py \
    --mask outputs/predictions/image_pred.png \
    --dem path/to/dem.tif \
    --output_dir outputs/gis
```

Outputs:
- Flow direction raster
- Flow accumulation raster
- Flow patterns within flooded areas

#### Export to GIS Formats
```python
from src.gis.export_utils import export_all_formats
from src.gis.export_utils import load_georeferencing_from_tiff

# Load georeferencing from source TIFF
georef = load_georeferencing_from_tiff('rawdata/source.tif')

# Export to multiple formats
export_all_formats(
    mask=prediction_mask,
    output_dir='outputs/gis',
    basename='flood_2021',
    transform=georef['transform'],
    crs=georef['crs'],
    formats=['geotiff', 'shapefile', 'geojson'],
    attributes={'date': '2021-09-02', 'confidence': 0.85}
)
```

## Configuration

### Training Configuration (`config/train_config.yaml`)

Key parameters:
```yaml
model:
  encoder: resnet34              # resnet34, efficientnet-b3, mobilenet_v2
  encoder_weights: imagenet      # Use pretrained weights

training:
  batch_size: 8                  # Adjust based on GPU memory
  epochs: 100
  learning_rate: 0.0001
  early_stopping_patience: 15

data:
  image_size: 512                # Input image size
  augmentation:                  # Strong augmentation for small dataset
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_limit: 30

training:
  loss:
    type: hybrid                 # bce, dice, focal, hybrid
    bce_weight: 0.5
    dice_weight: 0.5
```

### GIS Configuration (`config/gis_config.yaml`)

Key parameters:
```yaml
dem:
  source: USGS_3DEP              # DEM source
  resolution: 10                 # meters

extent:
  min_area_threshold: 100        # Minimum flood area (m²)

depth:
  method: terrain_based          # boundary_elevation or terrain_based
  min_depth_threshold: 0.1       # meters

export:
  formats:
    - geotiff
    - shapefile
    - geojson
```

## Dataset

### Current Dataset
- **Total samples**: 203 image-mask pairs
- **Source**: Planet Labs PS2 satellite + additional flood imagery
- **Resolution**: Variable (380x285 to 4859x3644 pixels)
- **Format**: JPEG images, PNG binary masks
- **Location**: Eastern USA (Pennsylvania/New Jersey area)
- **Date**: September 2, 2021

### Data Organization
- Images: `train/image/*.jpg`
- Masks: `train/mask/*.png`
- Metadata: `train/metadata.csv`
- Raw satellite: `rawdata/*.tif`

## Model Architecture

### U-Net with Pretrained Encoder

**Default Configuration:**
- Architecture: U-Net
- Encoder: ResNet34 (pretrained on ImageNet)
- Input: 512x512 RGB images
- Output: 512x512 binary mask (sigmoid activation)
- Parameters: ~21M

**Alternative Encoders:**
- ResNet50 (23M params) - Higher capacity
- EfficientNet-B3 (10M params) - Better accuracy
- MobileNetV2 (2M params) - Faster inference

See `MODELS.md` for detailed comparison.

## Performance Targets

### Model Metrics (POC)
- IoU > 0.60
- Dice > 0.70
- Pixel Accuracy > 0.85

### Training Time
- ~2-4 hours on T4 GPU (100 epochs)
- ~1-2 hours on A100 GPU

## Results Structure

```
outputs/
├── models/
│   ├── best_model.pth           # Best checkpoint (by val IoU)
│   └── last_checkpoint.pth      # Most recent checkpoint
├── logs/
│   ├── training_history.json    # Loss and metrics history
│   └── samples/                 # Validation sample images
├── predictions/
│   └── *.png                    # Predicted masks
└── gis/
    ├── extent_analysis.json     # Flood extent statistics
    ├── flood_depth.tif          # Depth map (GeoTIFF)
    ├── flood_2021.shp           # Shapefile export
    └── flood_2021.geojson       # GeoJSON export
```

## Jupyter Notebooks

Interactive notebooks for experimentation:

1. **`notebooks/01_data_exploration.ipynb`**
   - Dataset statistics and visualization
   - Image size distribution
   - Flood coverage analysis

2. **`notebooks/02_model_training.ipynb`**
   - End-to-end training workflow
   - Optimized for Google Colab
   - Includes Drive mounting and GPU setup

3. **`notebooks/03_gis_analysis.ipynb`**
   - Complete GIS workflow
   - Extent, flow, and depth analysis
   - Visualization and export

## Limitations & Future Work

### Current Limitations
- Small dataset (203 samples) limits model generalization
- No validation of depth/flow estimates against ground truth
- Flow direction requires DEM data acquisition
- Variable image sizes require careful preprocessing

### Future Enhancements
1. **Data Expansion**
   - Annotate additional tiles from raw satellite imagery
   - Integrate public datasets (Sen1Floods11, FloodNet)
   - Multi-temporal analysis (before/after flooding)

2. **Model Improvements**
   - Ensemble models for better accuracy
   - Multi-spectral input (use NIR band)
   - Test-time augmentation

3. **Deployment**
   - Web application for easy use
   - Real-time batch processing
   - Model optimization (ONNX, TensorRT)

## Citation

If you use this project, please cite:
```
@software{flood_estimation_2025,
  title = {Flood Estimation from Aerial Imagery},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

## License

[Add your license here]

## Acknowledgments

- Planet Labs for satellite imagery
- segmentation_models_pytorch library
- WhiteboxTools for terrain analysis
- USGS for DEM data

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email]

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [segmentation_models_pytorch Documentation](https://smp.readthedocs.io/)
- [WhiteboxTools User Manual](https://www.whiteboxgeo.com/manual/)
- [USGS 3DEP Program](https://www.usgs.gov/3d-elevation-program)
