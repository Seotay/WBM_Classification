# Robust Process Diagnosis using Explainable CNNs for Highly Imbalanced Wafer Map Defect Classification in Semiconductor Manufacturing

## Purpose
This repository implements a **CNN-based wafer bin map defect classification framework** for realistic semiconductor manufacturing data with severe class imbalance.

The project is based on the full labeled **WM-811K dataset (172,950 samples)** and keeps the dominant `None` class in validation and test sets to evaluate performance under production-like class distributions.


## Workflow

<p align="center">
  <img src="./figures/workflow.png" width="650"/>
  <br/>
  <em>Overall workflow of the CNN-based wafer defect classification framework.</em>
</p>


## Summary

- **Task**: 9-class wafer defect pattern classification
- **Dataset**: Full labeled WM-811K dataset, including 147,431 `None` samples
- **Architecture**: CNN Encoder + Multi-Layer Fully Connected Classifier
- **Input Representation**: 128 x 128 x 3 one-hot encoded wafer maps
- **Imbalance Strategy**: Failure-class geometric augmentation + class-weighted focal loss
- **Augmentation**: Horizontal flip, vertical flip, and random rotation applied only to defect classes
- **Evaluation**: 30 independent stratified train/validation/test splits
- **Best Performance**: Accuracy `0.976 +/- 0.002`, Macro-F1 `0.882 +/- 0.009`
- **Key Improvement**: Scratch recall improved from `0.388` to `0.725`
- **Framework**: PyTorch 2.4.0
- **GPU**: NVIDIA RTX 4080 16GB
- **Inference Latency**: `0.225 ms/sample` at batch size 512
- **Number of Classes**: 9  
  (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-Full`, `None`)

---

## Directory Structure

After downloading the WM-811K dataset, organize the project as follows:

```text
project/
  checkpoints/
    cnn_multi_class_best.pt
  data/
    WM-811K-labeled-dataset.pkl
  figures/
    workflow.png
  model/
    model.py
    TasiCNN.py
    ShinCNN.py
    JangCNN.py
    ChenCNN.py
    ChauhanCNN.py
    BiswasCNN.py
  utils/
    dataset.py
    loss.py
    trainer.py
    utils.py
  inference_latancy.py
  main.py
  README.md
```

### Notes

- Place the raw or preprocessed WM-811K labeled `.pkl` file inside the `data/` directory.
- Model checkpoints are saved in the `checkpoints/` directory.
- The full labeled dataset contains 172,950 samples across nine classes.
