# Pneumonia Detection from Chest X-Ray Images using Deep Ensemble Learning and Grad-CAM

## Overview

Pneumonia remains one of the leading causes of respiratory-related mortality worldwide. Early and accurate diagnosis from chest radiographs is critical for timely treatment and improved patient outcomes.

This project presents an **AI-powered desktop application** that automatically detects pneumonia from chest X-ray images using a **Deep Ensemble Learning framework** combining **DenseNet121** and **ResNet34**. To enhance transparency and clinical trust, the system integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, allowing users to visualize the image regions that influence the model's decision.

The application is built with **PyTorch**, accelerated with **CUDA**, and deployed through an intuitive **PyQt6 graphical interface** for real-time diagnostic assistance.

---

## Key Features

### Deep Ensemble Classification

* Combines predictions from **DenseNet121** and **ResNet34**
* Uses soft-voting ensemble strategy for improved robustness
* Reduces individual model bias and improves diagnostic reliability

### Explainable AI with Grad-CAM

* Generates visual attention heatmaps
* Highlights lung regions contributing to the prediction
* Improves interpretability and model transparency

### Clinical Decision Support

* Provides a simple binary output:

  * **YES** → Pneumonia Detected
  * **NO** → No Pneumonia Detected
* Designed for rapid screening and triage assistance

### Responsive Desktop Interface

* Built using **PyQt6**
* Uses **QThread-based asynchronous inference**
* Prevents UI freezing during model execution

### Automated Diagnostic Reports

* Stores Grad-CAM visualizations automatically
* Creates side-by-side comparison outputs for review and validation

---

## System Architecture

```text
Chest X-Ray Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 ┌──────────────────┐
 │   DenseNet121    │
 └──────────────────┘
        │
        ├──── Soft Voting Ensemble ────► Final Prediction
        │
 ┌──────────────────┐
 │    ResNet34      │
 └──────────────────┘
        │
        ▼
     Grad-CAM
        │
        ▼
 Heatmap Visualization
```

---

## Project Workflow

1. Upload a chest X-ray image.
2. Preprocess and normalize the image.
3. Perform inference using:

   * DenseNet121
   * ResNet34
4. Combine predictions through ensemble voting.
5. Generate Grad-CAM attention maps.
6. Display prediction and visualization results.
7. Save diagnostic outputs automatically.

---

## Sample Outputs

### Grad-CAM Diagnostic Visualization

<p align="center">
  <img src="diagnostic_outputs/cam_report_sample.png" width="850">
</p>

### Ensemble Confusion Matrix

<p align="center">
  <img src="diagnostic_outputs/confusion_matrix.png" width="500">
</p>

---

## Dataset Information

**Dataset:** Chest X-Ray Images (Pneumonia)

Source:

* Kaggle
* Mendeley Data

### Dataset Statistics

| Category   | Images |
| ---------- | -----: |
| Training   |  5,216 |
| Validation |    320 |
| Testing    |    320 |
| Total      |  5,856 |

### Classes

* Normal
* Pneumonia

Dataset Size: **~1.15 GB**

---

## Technology Stack

| Component               | Technology            |
| ----------------------- | --------------------- |
| Deep Learning Framework | PyTorch               |
| CNN Architectures       | DenseNet121, ResNet34 |
| Explainable AI          | Grad-CAM              |
| GUI Framework           | PyQt6                 |
| GPU Acceleration        | CUDA                  |
| Optimizer               | Adam                  |
| Loss Function           | CrossEntropyLoss      |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Darsh-Andharia/Pneumonia-Detection-Using-Cuda-Computing.git

cd Pneumonia-Detection-Using-Cuda-Computing
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Pre-trained Weights

Place the following model checkpoints in the project root directory:

```text
densenet_best.pt
resnet_best.pt
```

---

## Running the Application

Launch the desktop application:

```bash
python rapidrespiro_app.py
```

---

## How to Use

### Step 1: Upload Image

Click **Upload X-Ray Image** and select a supported image:

```text
.png
.jpg
.jpeg
```

### Step 2: Run Diagnosis

Click **Run Ensemble Diagnostic**.

### Step 3: Review Results

#### Positive Case

**YES – Pneumonia Detected**

* Red-highlighted result
* Grad-CAM heatmap generated
* Diagnostic report saved automatically

#### Negative Case

**NO – No Pneumonia Detected**

* Green-highlighted result
* No abnormal pneumonia patterns identified

---

## Future Improvements

* Multi-class thoracic disease classification
* Vision Transformer (ViT) integration
* Web-based deployment
* DICOM image support
* Real-time hospital PACS integration
* Model quantization for edge devices

---

## Project Structure

```text
Pneumonia-Detection-Using-Cuda-Computing/
│
├── rapidrespiro_app.py
├── densenet_best.pt
├── resnet_best.pt
├── requirements.txt
├── diagnostic_outputs/
│   ├── cam_report_sample.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## Authors

### Het Sachaniya

GitHub: https://github.com/Sachaniyahet

### Darsh Andharia

GitHub: https://github.com/DarshAndharia

---

## License

This project is intended for academic, educational, and research purposes. Clinical deployment should only be considered after extensive medical validation and regulatory approval.
