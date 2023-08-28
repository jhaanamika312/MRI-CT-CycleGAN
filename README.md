

# MRI to CT CycleGAN

This repository contains an implementation of a CycleGAN model to perform MRI to CT image translation. CycleGAN is a type of Generative Adversarial Network (GAN) that can learn to translate images from one domain to another without requiring paired data.

## Overview

The primary objective of this project is to do MRI-CT translation and vice versa and  used CycleGAn to demonstrate that.  This translation can be useful for various medical imaging applications, allowing for data augmentation, domain adaptation, or generating synthetic CT-like images from available MRI scans.

## Requirements

Before running the code, ensure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras (>=2.3)
- numpy
- matplotlib
- scikit-image
- other necessary libraries

## Dataset

I used the unpaired Dataset from Kaggle using this url https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan

## Usage

1. Prepare your MRI and CT datasets and organize them into appropriate folders (e.g., `images/trainA` for MRI images and `images/trainB` for CT images).

2. Adjust the hyperparameters and configuration in the provided code files as needed.

3. Train the CycleGAN model using the training data. You can use the provided training script by running:

   ```
   python train.py
   ```

4. After training, you can generate translated images from the trained model using the test dataset:

   ```
   python generate_images.py
   ```
5. The generated images will be saved in the specified output directory.

## Evaluation

In order to assess the quality of the generated images, the following evaluation metrics are calculated:

- **MSE (Mean Squared Error)**: Measures the average squared differences between the generated and actual CT images.
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the images by considering the ratio of the maximum possible signal power to the power of corrupting noise.
- **SSIM (Structural Similarity Index)**: Measures the structural similarity between the generated and actual CT images.

## Results

After generating the translated images, you can assess the quality of the results by examining the visualizations and evaluating the calculated metrics. Higher PSNR and SSIM values indicate a better quality of translated images.

## Credits

This project was inspired by the CycleGAN paper and utilizes various open-source libraries. Please refer to the relevant papers and repositories for more information.



Customize this README template to provide a clear and informative overview of your MRI to CT CycleGAN project. Include any additional information or sections that you find relevant to explain your code, methodology, and results.
