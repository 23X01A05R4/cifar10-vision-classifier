<div align="center">

# 🔍 CIFAR-10 Vision Classifier

### Image Classification using MobileNetV2 Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[📖 Documentation](documentation.md) • [🚀 Quick Start](#-quick-start) • [🌐 Web Demo](#-web-demo) • [📊 Results](#-results)

---

</div>

## 📌 Overview

This project implements an **image classification system** capable of recognizing **10 object categories** from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. It leverages **Transfer Learning** from a pre-trained **MobileNetV2** model (trained on ImageNet) and fine-tunes it for CIFAR-10 classification.

The project includes:
- 🧠 A **Jupyter Notebook** for model training with data augmentation
- 🖥️ A **CLI inference script** for classifying single images
- 🌐 A **Streamlit web app** for interactive, browser-based predictions

---

## 🗂️ Project Structure

```
cifar10-vision-classifier/
├── 📓 train.ipynb                   # Model training notebook
├── 🐍 inference.py                  # CLI inference script
├── 🌐 app.py                        # Streamlit web application
├── 🤖 cifar10_mobilenetv2.keras     # Pre-trained model weights
├── 📊 training_curves.png           # Training accuracy & loss plots
├── 🖼️  OIP.jpg                       # Sample test image
├── 📋 requirements.txt              # Python dependencies
└── 📖 documentation.md             # Detailed technical documentation
```

---

## 🏷️ Supported Classes

| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
|:-----------:|:-------------:|:-------:|:------:|:-------:|
| **🐶 Dog** | **🐸 Frog** | **🐴 Horse** | **🚢 Ship** | **🚛 Truck** |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/23X01A05R4/cifar10-vision-classifier.git
cd cifar10-vision-classifier
```

### 2. Set up the environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run inference (CLI)

```bash
python inference.py --image_path OIP.jpg
```

**Example Output:**
```
========================================
  Results for: OIP.jpg
========================================
  #1  ship         ███████████████████░  99.09% ← TOP PICK
  #2  airplane     ░░░░░░░░░░░░░░░░░░░░   0.74%
  #3  truck        ░░░░░░░░░░░░░░░░░░░░   0.09%
========================================
```

---

## 🌐 Web Demo

Launch the interactive Streamlit web app:

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

Features:
- 📤 Drag & drop image upload
- 🏆 Top prediction with confidence score
- 📊 Interactive bar chart for all 10 class probabilities

---

## 🧠 Model Architecture

```
Input (32×32×3)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
Resize (96×96)
    ↓
MobileNetV2 Preprocessing
    ↓
MobileNetV2 Base (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.2)
    ↓
Dense (10, softmax) → Output
```

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 (ImageNet) |
| Input Shape | 32 × 32 × 3 |
| Internal Resize | 96 × 96 |
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Batch Size | 64 |
| Epochs | 5 |

---

## 📊 Results

![Training Curves](training_curves.png)

The training curves show **accuracy and loss** over 5 epochs for both training and validation sets.

---

## 📦 Dependencies

```
tensorflow>=2.21
matplotlib
numpy
pillow
streamlit
plotly
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📖 Documentation

For detailed technical documentation including full training code, inference walkthrough, and architecture explanation, see [documentation.md](documentation.md).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

<div align="center">

Made with ❤️ using TensorFlow & Streamlit

⭐ **Star this repo if you found it helpful!**

</div>
