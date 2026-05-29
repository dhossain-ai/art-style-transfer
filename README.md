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