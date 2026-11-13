# Leaf Disease Segmentation

The model performs binary semantic segmentation: every image pixel is classified as healthy/background or diseased.


## Dataset

The project uses the [Leaf Disease Segmentation Dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset), containing 588 original leaf images and matching masks.

The preparation notebook creates a deterministic split using seed `42`:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

Training augmentation is applied online and never to validation or test images.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab
```

## Usage

### 1. Prepare the data

Run `notebooks/data_preparation_and_eda.ipynb` from beginning to end.

It performs:

- Image-mask matching
- Basic exploratory data analysis
- Disease-region visualization
- A 70/15/15 train-validation-test split
- Resizing to `256 × 256`
- Binary mask conversion
- Saving individual processed images and masks under `data/processed/`

### 2. Train the model

Run `notebooks/model_training.ipynb` after the processed folders have been created.

The training pipeline uses:

- ImageNet-pretrained ResNet50 encoder
- U-Net-style decoder and skip connections
- Group normalization in the decoder
- Horizontal and vertical flips, rotation, scaling, and brightness augmentation
- Dice plus focal loss
- Adam optimizer
- Frozen-encoder training followed by fine-tuning
- Early stopping and learning-rate reduction based on validation IoU


## Evaluation

The notebook reports and plots:

- Total loss
- Dice loss
- Pixel accuracy
- Dice coefficient
- Intersection over Union (IoU)

The prediction threshold is selected using validation data only. That threshold is then fixed before the final test evaluation.


## Model outputs

Training creates checkpoints under `models/`:

```text
models/resnet50_unet_frozen.keras
models/best_resnet50_unet.keras
```

The final notebook section displays the original test image, ground-truth mask, and predicted mask side by side.

