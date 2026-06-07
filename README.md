Here is the updated, complete `README.md` file restructured to match the exact schema, formatting, and layout of your AES File Encryptor example, while containing your pneumonia project's details.

### `README.md`

```markdown
# Pneumonia Detection from Chest X-Ray Images Using Deep Ensemble Learning & Grad-CAM

## Overview
Pneumonia Detection from Chest X-Ray Images is a GUI-based desktop application that utilizes deep learning ensemble techniques to identify pneumonia abnormalities in digital radiographs. The core system blends predictions from DenseNet121 and ResNet34 models, while integrating an optimized Grad-CAM framework to generate visual attention maps identifying critical diagnostic regions of interest.

## Features
- **Ensemble-Driven Diagnostic Matrix**: Evaluates scans using soft-voting predictions across both DenseNet121 and ResNet34.
- **Explainable AI Integration (Grad-CAM)**: Captures features and gradients dynamically from final convolutional layers using isolated tensor cloning to avoid backend memory conflicts.
- **Binary Clinical Readout**: Simplifies raw probability output into a rapid, high-contrast "YES" or "NO" classification for fast clinical triage.
- **Multi-Threaded UI Performance**: Runs asynchronous network passes inside a detached execution worker thread (`QThread`), keeping the PyQt6 desktop framework fluid and responsive.
- **Automated Verification Reports**: Automatically matches positive scans with side-by-side verification images inside an output subdirectory.

## Pipeline Deliverables
### Live Grad-CAM Generation Analytics:
<kbd>
  <img src="diagnostic_outputs/cam_report_sample.png" alt="Grad-CAM Output Pipeline Preview" width="800px">
</kbd>

### Ensemble Validation Matrix:
<kbd>
  <img src="diagnostic_outputs/confusion_matrix.png" alt="Ensemble Confusion Matrix Diagram" width="500px">
</kbd>

## Dataset Details
- **Dataset Source**: Chest X-Ray Images (Pneumonia) [Kaggle / Mendeley Data]
- **Target Profiles**: 2 Balance Output Categories (Normal / Pneumonia)
- **Data Footprint**: 5,856 Total Diagnostic Images (1.15 GB Total Volume)
  - **Training Set**: 5,216 Images
  - **Validation Set**: 320 Images
  - **Testing Set**: 320 Images

## System Core Parameters
- **Compute Architecture**: PyTorch Engine (CUDA Hardware Acceleration Ready)
- **Backbone Framework Modules**: DenseNet121 + ResNet34 Base Deployments
- **Optimizer & Objective Criteria**: Adam Optimizer paired with CrossEntropyLoss
- **Graphical GUI Environment**: PyQt6 (Configured with high-DPI automatic desktop scaling)

## Installation
1. Clone this repository:
   ```sh
   git clone [https://github.com/Darsh-Andharia/Pneumonia-Detection-Using-Cuda-Computing.git](https://github.com/Darsh-Andharia/Pneumonia-Detection-Using-Cuda-Computing.git)
   cd Pneumonia-Detection-Using-Cuda-Computing

```

2. Install the required deep learning and desktop environment dependencies:
```sh
pip install -r requirements.txt

```


3. Place your pre-trained PyTorch weight checkpoints in the project root directory:
* Ensure `densenet_best.pt` and `resnet_best.pt` match the target structural classes setup.



## Usage

### Running the Main GUI Console

```sh
python rapidrespiro_app.py

```

### Run Diagnostic Scan:

1. Click on **"Upload X-Ray Image"** to choose your digital radiograph (`.png`, `.jpg`, `.jpeg`).
2. Once verified in the main viewer canvas window, press **"Run Ensemble Diagnostic"**.
3. The application will compute prediction matrices and render the diagnostic determination:
* **YES**: (Highlighted in warning crimson) Pneumonia detected. A side-by-side Grad-CAM heatmap localization layer maps to your screen and archives in the output folder.
* **NO**: (Highlighted in medical green) No target abnormalities matching clinical criteria detected.



## Contribution

Contributions are welcome! Feel free to submit issues and pull requests to upgrade structural layer backbones or test wider multi-class variants.

## Authors

* **[Sachaniya Het](https://github.com/Sachaniyahet)**
* **[Darsh Andharia](https://github.com/DarshAndharia)**

```

```
