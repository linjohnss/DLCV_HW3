# HW3 · Instance Segmentation of Medical Cells

## Overview
This project implements an enhanced **Mask R‑CNN** pipeline for instance segmentation on a colored‑cell microscopy dataset.  
Key components:

* **ConvNeXt‑Base** backbone  
* **Deep mask head** (6 conv layers)  
* Lightweight **Transformer decoder** for global context  
* **Focal Loss** to address class imbalance  

The goal is to maximise mAP on the Codabench leaderboard while keeping the model under 200 M parameters.

---

## Requirements
* Python ≥ 3.9  
* PyTorch ≥ 2.0  
* CUDA‑enabled GPU (≥ 11 GB for batch 2)  
* Additional packages in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Dataset
Place the files as below after downloading from the course link:
```
data/
 ├─ train/                 # 209 folders, each with image.tif + class?.tif
 ├─ test_release/          # 101 test images
 └─ test_image_name_to_ids.json
```

## Usage

### Training
```bash
python main.py \
  --data_dir data \
  --backbone convnext_base \
  --batch_size 2 \
  --epochs 40 \
  --use_transformer \
  --mask_head_type deep \
  --output_dir runs/hw3
```

### Testing
```bash
python main.py \
  --data_dir data \
  --eval \
  --ckpt_name runs/hw3/best.pth
```

### Core Arguments
| Flag                | Description                               | Default         |
| ------------------- | ----------------------------------------- | --------------- |
| `--backbone`        | Backbone (`resnet50`, `convnext_base`, …) | `convnext_base` |
| `--epochs`          | Training epochs                           | 40              |
| `--batch_size`      | Images per GPU                            | 2               |
| `--lr`              | Base learning rate                        | `3e-5`          |
| `--use_transformer` | Enable transformer decoder                | false           |
| `--mask_head_type`  | `default`, `deep`, `wider`                | `deep`          |
| `--eval`            | Run test‑time inference                   | false           |

View all options:
```bash
python main.py -h
```

## Project Structure
- `main.py`: Main training and testing script
- `models/`: Contains model definitions
  - `mask_rcnn.py`: Custom Mask R‑CNN (ConvNeXt + Transformer)
- `utils/`: Utility functions
  - `dataloader.py`: Dataset, augmentations, RLE utilities
  - `trainer.py`: Training loop, validation, visualisation

## Performance snapshot

![image](leaderboard.png)