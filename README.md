# Brain Tumor Segmentation from MRI Scans

This project implements an automated brain tumor segmentation system using multi-modal MRI scans from the BraTS dataset and a multi-class U-Net model.

## Features

- Loads BraTS MRI modalities: T1, T1ce, T2, FLAIR
- Preprocesses MRI slices by resizing and normalization
- Performs multi-class tumor sub-region segmentation
- Detects:
  - Necrotic core
  - Edema
  - Enhancing tumor
- Trains a U-Net deep learning model
- Evaluates predictions using Dice, IoU, and Accuracy
- Visualizes overlays of predicted tumor regions on MRI scans
- Saves reports, metrics, plots, and prediction outputs
- Provides a Streamlit-based graphical interface

## Project Structure

- `src/` contains core source code
- `outputs/` stores models, plots, logs, reports, and prediction images
- `dataset/` contains BraTS dataset
- `main.py` runs the full training and evaluation pipeline
- `ui.py` launches the Streamlit interface

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt