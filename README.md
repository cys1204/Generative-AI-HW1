# GAN Assignment Toolkit

This repository provides reference implementations for three distinct GAN-based tasks aligned with the homework specification:

1. **Face Generation** using a DCGAN trained on CelebA.
2. **Image-to-Image Translation** with CycleGAN on horse2zebra or other unpaired datasets.
3. **Image Super Resolution** leveraging a WGAN objective inspired by SRGAN for CIFAR-10 upscaling.

Each project lives in its own folder under `GAN_Assignment/` with dedicated training and evaluation utilities.

```
GAN_Assignment/
├── face_generation_dcgan/
│   ├── model.py          # DCGAN generator and discriminator definitions
│   ├── train.py          # CelebA training script with checkpointing and sample export
│   ├── generate.py       # Utility to sample images from a trained generator
│   └── data/celeba/      # Placeholder for CelebA dataset (use torchvision ImageFolder layout)
├── image_translation_cyclegan/
│   ├── model/
│   │   ├── generator.py  # ResNet-based generator module
│   │   └── discriminator.py  # PatchGAN discriminator
│   ├── train.py          # CycleGAN training loop with cycle & identity losses
│   ├── test.py           # Inference helper to translate folders of images
│   └── data/horse2zebra/ # Placeholder for horse2zebra dataset (trainA/trainB folders)
└── super_resolution_wgan/
    ├── model.py          # SR generator and critic definitions
    ├── train.py          # WGAN-GP style training on CIFAR-10
    ├── evaluate.py       # PSNR/SSIM evaluation pipeline
    └── data/cifar10/     # Placeholder for CIFAR-10 dataset
```

## Getting Started

1. Create the expected dataset directories under each task folder and place or download the corresponding datasets. The provided scripts rely on the standard `torchvision.datasets` structures.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training scripts. Example commands:

   ```bash
   # Train DCGAN on CelebA
   python -m GAN_Assignment.face_generation_dcgan.train /path/to/celeba --output-dir checkpoints/dcgan

   # Train CycleGAN on horse2zebra
   python -m GAN_Assignment.image_translation_cyclegan.train /path/to/horse2zebra --output-dir checkpoints/cyclegan

   # Train WGAN super-resolution on CIFAR-10
   python -m GAN_Assignment.super_resolution_wgan.train GAN_Assignment/super_resolution_wgan/data/cifar10 --output-dir checkpoints/wgan_sr
   ```

4. Evaluate or generate outputs using the corresponding helper scripts (`generate.py`, `test.py`, `evaluate.py`).

These modules can be extended with additional logging, evaluation metrics (e.g., FID, IS), or parameter studies to support the written report required for the homework submission.
