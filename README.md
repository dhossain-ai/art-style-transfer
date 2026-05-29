# 🎨 Art Style Transfer

> Applying famous art styles to personal photos using Neural Networks (VGG19) and Classical Image Processing methods.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-RTX%204060-green?style=flat&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Overview

This project implements **Neural Style Transfer** using a pre-trained VGG19 convolutional neural network, alongside **Classical Style Transfer** methods for comparison. A full **Tkinter desktop app** is included for interactive use.

The project was built as part of the **Image Recognition Systems** university course.

---

## 🖼️ Results

### Neural Style Transfer (VGG19)

| Content | Van Gogh | Monet | Picasso | Warhol |
|---------|----------|-------|---------|--------|
| ![content](images/content/my_photo.jpg) | ![vangogh](images/output/result_vangogh.jpg) | ![monet](images/output/result_monet.jpg) | ![picasso](images/output/result_picasso.jpg) | ![warhol](images/output/result_warhol.jpg) |

### Classical Methods

| Content | Reinhard | Histogram | Sketch |
|---------|----------|-----------|--------|
| ![content](images/content/my_photo.jpg) | ![reinhard](images/output/result_classical_reinhard.jpg) | ![histogram](images/output/result_classical_histogram.jpg) | ![sketch](images/output/result_classical_sketch.jpg) |

---

## 🧠 How It Works

### Neural Method (VGG19)

```text
Content Image ──→ ┐
                  VGG19 (frozen) ──→ Extract Features
Style Image   ──→ ┘
                          ↓
               Optimize new image to match:
               • Content features (structure)
               • Style features via Gram Matrix (textures)
                          ↓
                    Stylized Result ✅
```

VGG19 is used as a **frozen feature extractor** — its weights are never modified. Only the output image is optimized.

**Content layers:** `conv4_2` — captures structure and objects

**Style layers:** `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1` — captures textures at multiple scales

### Classical Methods

| Method | How it works | Needs Style Image? |
|--------|-------------|-------------------|
| **Reinhard Color Transfer** | Transfers color mean & std in LAB color space | ✅ Yes |
| **Histogram Matching** | Matches color distribution channel by channel | ✅ Yes |
| **Pencil Sketch** | Edge-based stylization using Gaussian blur | ❌ No |

### Neural vs Classical — Key Difference

| | Classical | Neural |
|--|-----------|--------|
| What it transfers | Color statistics only | Colors + textures + brushstrokes |
| Result feels like | Color-filtered photo | Actual painting |
| Speed | Instant | 3–6 min (GPU) |
| Requires deep learning | ❌ No | ✅ Yes |

---

## 🖥️ Desktop App

The project includes a full **Tkinter GUI app** (`src/app.py`):

- 📸 Browse and select content image
- 🖼️ Choose preset art style or browse custom
- ⚙️ Switch between Neural and Classical methods
- 📊 Live progress bar during processing
- 💾 Save result with one click

---

## 📁 Project Structure

```text
art-style-transfer/
├── images/
│   ├── content/                  # Input photo
│   ├── style/                    # Art style paintings
│   └── output/                   # Generated results
├── src/
│   ├── config.py                 # Settings and paths
│   ├── model.py                  # VGG19 loader and feature extractor
│   ├── utils.py                  # Image loading and saving utilities
│   ├── style_transfer.py         # Neural style transfer algorithm
│   ├── classical.py              # Classical methods
│   ├── main.py                   # Command line runner
│   └── app.py                    # Tkinter desktop app
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/art-style-transfer.git
cd art-style-transfer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your images

```text
images/content/my_photo.jpg         ← your photo
images/style/vangogh_starry_night.jpg
images/style/monet_water_lilies.jpg
images/style/picasso_cubism.jpg
images/style/warhol_popart.jpg
```

---

## 🚀 Usage

### Option 1 — Desktop App (Recommended)

```bash
cd src
python app.py
```

### Option 2 — Command Line (all styles)

```bash
cd src
python main.py
```

### Option 3 — Command Line (single style)

```bash
cd src
python main.py vangogh
python main.py monet
python main.py picasso
python main.py warhol
```

---

## 🔧 Configuration

Edit `src/config.py` to tune the style transfer:

```python
NUM_STEPS      = 600       # more steps = stronger style
STYLE_WEIGHT   = 5000000   # higher = more artistic
CONTENT_WEIGHT = 1         # higher = more photo-like
LEARNING_RATE  = 0.01      # optimization speed
IMAGE_SIZE     = 512       # reduce to 256 if GPU memory issues
```

---

## 📦 Requirements

```text
torch
torchvision
pillow
matplotlib
opencv-python
```

> VGG19 weights (~548MB) are downloaded automatically on first run from PyTorch Hub.

---

## 💻 Hardware

| Device | Time per style |
|--------|---------------|
| RTX 4060 (GPU) | ~3–6 min |
| CPU only | ~30–60 min |

---

## 📚 References

- Gatys et al. (2016) — [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Reinhard et al. (2001) — Color Transfer between Images
- VGG19 — Simonyan & Zisserman, Oxford Visual Geometry Group

---

## 👨‍💻 Author

Built for **Image Recognition Systems** university course.