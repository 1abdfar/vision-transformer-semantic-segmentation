# Vision Transformer for Semantic Segmentation

This repository contains an end-to-end Machine Learning pipeline utilizing a **Vision Transformer (ViT)** backbone for dense pixel prediction (Semantic Segmentation). The project is built using PyTorch and trained on the [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset.


## Project Architecture

1. **Encoder (Feature Extraction):** A pre-trained `ViT-B/32` model is used to extract spatial context. Input images are tokenized into non-overlapping patches, embedded, and passed through consecutive self-attention blocks.
2. **Decoder (Dense Prediction):** The sequence of output embeddings is reshaped back into a 2D spatial feature map. A custom transpose-convolution (UpConv) head gradually upsamples the latent representation to construct the final pixel-level classification mask.


## Dataset

This project leverages the **ADE20K** dataset, a large-scale semantic segmentation dataset providing dense annotations for 150 distinct semantic categories (e.g., sky, building, person, car). 

## Repository Structure

* `dataset.py` - PyTorch `Dataset` wrappers and interpolation-safe data transforms.
* `model.py` - Scratch implementations of Multi-Head Attention, Transformer Blocks, and the `SegmentationViT` composite model.
* `train.py` - Primary optimization loop, loss computation, and checkpointing logic.
* `inference.py` - Utilities for running forward passes on single images and visualizing segmentation maps side-by-side.

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision
* Einops, Pillow, NumPy, Matplotlib, tqdm

### Installation & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/vision-transformer-semantic-segmentation.git](https://github.com/yourusername/vision-transformer-semantic-segmentation.git)
   cd vision-transformer-semantic-segmentation