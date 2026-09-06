# Wafer Defect Classification

## Purpose
This repository implements a **CNN-based wafer defect pattern classification framework** designed for severe class imbalance conditions in real semiconductor manufacturing environments.

The model is evaluated on the **full labeled WM-811K dataset (172,950 samples)** without excluding wafers with irregular resolutions or aspect ratios.


## Workflow

<p align="center">
  <img src="./figures/workflow.jpg" width="650"/>
  <br/>
  <em>Overall workflow of the proposed CNN-based wafer defect classification framework.</em>
</p>


## Summary

- **Architecture**: CNN Encoder + Multi-Layer Fully Connected Classifier
- **Input Resolution**: 128 × 128 (aspect-ratio preserved resizing)
- **Loss Function**: Focal Loss (α = 0.15, γ ∈ {1.5, 2.5, 3.5})
- **Data Augmentation**: Applied to failure-pattern classes only
- **Number of Classes**: 9  
  (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-Full`, `None`)
- **Framework**: PyTorch 2.4.0
- **GPU**: NVIDIA RTX 4080 (16GB)

---

## Directory Structure
After downloading the WM-811K Dataset, files in the following directory structure:

```
project/
    ├── config/
    │   └── config.py  # Training & Inference configuration
    │
    ├── checkpoints/
    │   └── cnn_multi_class_best.pt
    │
    ├── utils/
    │   ├── dataset.py
    │   ├── utils.py
    │
    ├── data/
    │   ├── README.md
    │   ├── WM-811K-labeled dataset(.pkl)       
    │
    ├── main.py         
    ├── model.py          
    └── trainer.py                     

### Notes
- The raw WM-811K dataset should be placed inside the `data/` directory.
- The `.pkl` file contains only the labeled portion (172,950 samples).
- Model checkpoints are saved in the `checkpoints/` directory.
```