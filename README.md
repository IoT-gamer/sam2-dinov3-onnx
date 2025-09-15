# DINOv3 and EdgeTAM (SAM2) with ONNX Runtime

This repository provides a set of tools and examples for converting and utilizing powerful vision models, DINOv3 and EdgeTAM (SAM2), within the ONNX ecosystem. The focus is on creating efficient, PyTorch-independent inference pipelines for tasks like one-shot segmentation, foreground extraction, and robust video object tracking.

## 📂 Repository Structure
```bash
├── notebooks/
│   ├── dinov3_onnx_export.ipynb               # Exports DINOv3 to ONNX
│   ├── edgetam_onnx_export.ipynb              # Exports EdgeTAM encoder/decoder to ONNX
│   ├── foreground_segmentation_onnx_export.ipynb # Trains and exports a foreground classifier
│   └── dinov3_one_shot_segmentation_onnx.ipynb  # Demo for one-shot segmentation
│
└── scripts/
    └── hybrid_tracker.py   
```

## 📓 Notebooks

Each notebook is self-contained and can be run directly in Google Colab.

| Notebook | Description | Link |
| :--- | :---: | ---: |
| `dinov3_onnx_export.ipynb` | Converts the DINOv3 Vision Transformer (ViT) feature extractor to ONNX format. | [link](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/tree/main/notebooks/dinov3_onnx_export.ipynb) |
| `edgetam_onnx_export.ipynb` | Exports the EdgeTAM image encoder and mask decoder models to ONNX for efficient segmentation. | [link](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/tree/main/notebooks/edgetam_onnx_export.ipynb) |
| `foreground_segmentation_onnx_export.ipynb` | Trains a logistic regression classifier on DINOv3 features for foreground segmentation and exports it to ONNX | [link](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/tree/main/notebooks/foreground_segmentation_onnx_export.ipynb) |
| `dinov3_one_shot_segmentation_onnx.ipynb` | Demonstrates one-shot segmentation using DINOv3 features and a reference mask, all in ONNX. | [link](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/tree/main/notebooks/dinov3_one_shot_segmentation_onnx.ipynb) |

## 🙏 Acknowledgements
This work builds upon the official implementations and research from the following projects:

**DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

**EdgeTAM:** [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)

**Space-Time Correspondence as a Contrastive Random Walk:** [ajabri/videowalk](https://github.com/ajabri/videowalk)