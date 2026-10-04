# Underwater Image Enhancement using Deep Learning (EUVP)

A PyTorch project that enhances blurry, low-contrast, color-distorted underwater images using a convolutional encoder-decoder network trained on the EUVP dataset.

## Problem
Underwater photos suffer from color cast, low contrast and haze because water absorbs and scatters light. This project trains a model to map a degraded underwater image to an enhanced version.

## Dataset
- **EUVP (Enhancing Underwater Visual Perception)**, Paired subset (`underwater_imagenet`)
- 3,700 paired images (degraded input in `trainA`, enhanced ground truth in `trainB`)
- Source: Kaggle (`pamuduranasinghe/euvp-dataset`), CC0 licence

## Approach
1. Load paired images and resize them to 256×256.
2. Build a simple convolutional encoder-decoder (U-Net style) in PyTorch.
3. Train with **L1 loss** and the **Adam** optimizer (lr = 0.001, batch size 8, 3 epochs) on a Tesla T4 GPU in Google Colab.
4. Save the weights (`unet_euvp.pth`) and run inference on custom test images.

## Results
| Metric | Value |
|--------|-------|
| Training loss (L1) | 0.0672 → 0.0564 over 3 epochs |
| PSNR (one training batch, output vs ground truth) | ≈ 22.6 dB |
| SSIM (one test image, original vs enhanced) | ≈ 0.79 |

The notebook shows side-by-side comparisons of the input, the enhanced output and the ground truth.

## Tech Stack
Python, PyTorch, torchvision, Pillow, matplotlib, scikit-image, Google Colab

## How to Run
1. Open `EUVP_Underwater_image_Enhancement.ipynb` in Google Colab (GPU runtime).
2. Upload your `kaggle.json` to download the dataset.
3. Run all cells to train the model, or load the saved `unet_euvp.pth` to enhance your own images.

## Limitations and Future Work
- The model is small and trained for only 3 epochs, so the enhancement is modest.
- PSNR and SSIM were measured on a single batch or image, not on a full validation set.
- Possible improvements: a deeper U-Net with skip connections, more epochs, perceptual or adversarial (GAN) loss, and evaluation on the full test split.
