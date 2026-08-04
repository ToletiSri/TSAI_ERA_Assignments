# Session 13 — YOLOv3 Object Detection

This assignment ports a YOLOv3 training pipeline to PyTorch Lightning and trains it for multi-class object detection.

## Core concepts

- **YOLOv3 architecture:** residual feature extraction and predictions at three scales for objects of different sizes.
- **Anchor-based detection:** each grid cell predicts box offsets, objectness, and class scores relative to predefined anchors.
- **Detection loss:** combines bounding-box regression, object/no-object confidence, and classification losses.
- **Data augmentation:** Albumentations transforms and mosaic augmentation expand the variety of training scenes.
- **Mixed-precision training:** Lightning and 16-bit arithmetic reduce memory use and speed up training.
- **Detection metrics and post-processing:** intersection over union (IoU), non-maximum suppression (NMS), class/objectness accuracy, and mean average precision (mAP).
- **Deployment and interpretability:** utilities visualize detections and support a Hugging Face demo workflow.

## Implementation map

- `model.py` defines YOLOv3 blocks and multi-scale prediction heads.
- `dataset.py` prepares images, bounding boxes, anchors, and mosaic samples.
- `loss.py` implements the YOLO loss.
- `LightningModel.py` contains the Lightning training and validation workflow with One Cycle learning-rate scheduling.
- `utils.py` provides IoU, NMS, mAP, checkpointing, and visualization helpers.
- `config.py` centralizes anchors, image sizes, transforms, and training settings.
- `S13.ipynb` runs the experiment.

## Recorded results

| Split | Class accuracy | No-object accuracy | Object accuracy |
| --- | ---: | ---: | ---: |
| Train | 82.71% | 98.51% | 63.68% |
| Test | 79.26% | 98.68% | 55.93% |

The associated [Hugging Face Space](https://huggingface.co/spaces/ToletiSri/TSAI_S13) accepts images for inference.
