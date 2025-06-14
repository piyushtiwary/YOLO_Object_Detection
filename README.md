# 🧠 YOLO Object Detection

A TensorFlow-based implementation of the **YOLO (You Only Look Once)** algorithm for real-time object detection on images. This project demonstrates training on the Pascal VOC 2012 dataset and inference on COCO sample images.

---

## 📌 Overview

This project demonstrates how to build a YOLO-style object detection model from scratch using TensorFlow/Keras. The implementation includes:

- Custom YOLO architecture tailored for bounding box prediction and classification
- Training pipeline using the **Pascal VOC 2012** dataset
- Inference and visualization on **COCO** dataset images
- Custom grid-based output decoding and visualization logic

---

## 🚀 YOLO Algorithm – How It Works

YOLO (You Only Look Once) is a single-shot object detection algorithm that divides an image into a grid and for each grid cell:

- Predicts **whether an object is present**
- Predicts **bounding box coordinates** (x, y, width, height) relative to the cell
- Predicts **class probabilities** over all object classes

Instead of running region proposals + classifiers (as in R-CNN family), YOLO performs all computations in a single forward pass of a convolutional neural network, making it extremely fast and efficient.

---

## 🗂️ Dataset

### ✅ Training: **Pascal VOC 2012**
- Contains 20 object classes: person, car, dog, etc.
- Annotations in XML format, parsed and converted to YOLO’s grid-based format
- Preprocessing includes resizing to `224x224`, normalizing pixel values, and encoding bounding boxes into a custom `(S, S, 5 + C)` tensor format

### 🧪 Testing: **COCO Sample Images**
- Used for inference and demonstration
- Images resized to `224x224` and passed through the trained model for prediction

---

## 📈 Model Configuration

- Image size: `224 x 224`
- Grid size: `S = H // 32 = 7` (7×7 grid)
- Classes: `20` (from Pascal VOC)
- Output tensor shape: `(S, S, 5 + C)` = `(7, 7, 25)`
  - 1 objectness score
  - 4 bounding box coordinates
  - 20 class probabilities

---

## 🛠 Features

- ✅ Full training pipeline with custom data loader and encoder for Pascal VOC annotations
- ✅ YOLO-style prediction tensor decoding 
- ✅ Grid overlay with per-cell predictions
- ✅ Utility for viewing predictions directly on COCO images
- ✅ TensorFlow/Keras-based implementation (easy to extend)

---

## 💡 Future Work
 Support multiple anchor boxes (YOLOv2+)

 Real-time webcam inference

 mAP (mean average precision) evaluation

 Integration with COCO-trained models

---

## 📚 References
  YOLO: https://pjreddie.com/darknet/yolo/
  
  Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
  
  COCO Dataset: https://cocodataset.org/

