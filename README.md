# Handwritten Digit Recognition

A Convolutional Neural Network (CNN) that classifies handwritten digits (0–9) from the **MNIST** dataset. Built with TensorFlow / Keras.

---

## Overview

| | |
|---|---|
| **Task** | Multi-class image classification (10 classes) |
| **Dataset** | MNIST — 70,000 grayscale 28×28 images |
| **Model** | Convolutional Neural Network |
| **Framework** | TensorFlow / Keras |
| **Trained model** | [`hdr.h5`](./hdr.h5) |

---

## Architecture

A compact CNN suited to MNIST-scale inputs:

- **Conv2D + ReLU** — feature extraction
- **MaxPooling2D** — spatial down-sampling
- **Conv2D + ReLU** — deeper features
- **Flatten → Dense (ReLU) → Dropout** — classifier head
- **Dense (Softmax, 10 units)** — class probabilities

Loss: `sparse_categorical_crossentropy` · Optimizer: `Adam` · Metric: `accuracy`

---

## Preprocessing

1. Load MNIST via `tensorflow.keras.datasets.mnist.load_data()`
2. Reshape `(28, 28)` → `(28, 28, 1)` to add a channel dimension
3. Normalize pixel values to `[0, 1]`
4. Optional: one-hot encode labels for categorical training (or use sparse labels directly)

---

## Tech Stack

- **Python**, **TensorFlow / Keras**
- **NumPy**, **Matplotlib**
- **Pillow** (image I/O for custom inference)

---

## How to Run

```bash
pip install tensorflow numpy matplotlib pillow
```

Open the notebook and run all cells:

```bash
jupyter notebook handwritten_digit_recognition.ipynb
```

To load and use the trained model directly:

```python
from tensorflow.keras.models import load_model
model = load_model("hdr.h5")
prediction = model.predict(your_28x28_grayscale_image_reshaped_to_1x28x28x1)
```

---

## Repository Contents

```
.
├── handwritten_digit_recognition.ipynb   # Training + evaluation notebook
├── hdr.h5                                # Trained Keras model
├── image.png                             # Sample / demo image
└── README.md
```

---

## License

MIT
