# AstroShield – AI Vision for Safety in Space

## Overview
AstroShield is an AI-powered real-time object detection system developed during the **HackWithHyderabad Hackathon**. The project is designed to improve astronaut safety inside space stations by continuously monitoring mission-critical safety equipment using the YOLOv8 object detection model.

The system detects essential emergency objects such as oxygen tanks, fire extinguishers, first aid kits, emergency phones, and safety panels, ensuring that astronauts can quickly locate life-saving equipment during critical situations.

**Team Name:** Amaravati Avengers

**Project Title:** AI-Powered Safety Object Detection in Space Stations

**Tagline:** *Enhancing astronaut safety through real-time intelligent monitoring.*

---

## Features

- Real-time detection of mission-critical safety equipment
- Detects 7 essential safety object classes
- High-speed inference (25 FPS on standard GPU)
- Accurate object localization using YOLOv8
- Supports images, videos, and live camera streams
- Lightweight deployment using ONNX optimization
- Easily extendable with additional safety object classes

---

## Safety Objects Detected

- Oxygen Tank
- Nitrogen Tank
- First Aid Box
- Fire Alarm
- Safety Switch Panel
- Emergency Phone
- Fire Extinguisher

---

# Methodology

## 1. Data Preparation

A custom dataset was created and manually annotated for seven mission-critical safety objects commonly found inside space stations.

To improve model generalization, extensive preprocessing and augmentation techniques were applied:

- Image resizing
- Normalization
- Rotation
- Horizontal flipping
- Brightness variation

These augmentations helped the model perform reliably under varying lighting conditions and viewing angles.

---

## 2. Model Selection

We selected **YOLOv8** because it offers an excellent balance between detection accuracy and real-time inference speed, making it ideal for astronaut safety monitoring.

The model was fine-tuned using pretrained COCO weights to leverage transfer learning while adapting to our custom dataset.

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8 |
| Epochs | 50 |
| Batch Size | 8 |
| Image Size | 640 × 640 |
| Optimizer | SGD |
| Learning Rate | Adaptive Scheduling |
| Initialization | COCO Pretrained Weights |

---

## 4. Evaluation Metrics

Model performance was evaluated using standard object detection metrics:

- Precision
- Recall
- F1 Score
- mAP@50
- mAP@50–95

---

# Results

| Metric | Score |
|---------|--------|
| mAP@50 | **87.8%** |
| mAP@50-95 | **78.9%** |
| Precision | **92.3%** |
| Recall | **83.4%** |
| F1 Score | **87.5%** |
| Inference Speed | **25 FPS** |

### Performance Highlights

- High detection accuracy across all safety equipment classes
- Minimal confusion between classes
- Slight overlap observed only between Oxygen Tank and Nitrogen Tank
- Bounding box confidence consistently above 0.80
- Suitable for real-time deployment

---

# Challenges

### 1. Limited Dataset

Only a small number of annotated samples were available for each object class.

**Solution**

- Applied aggressive data augmentation
- Used class-weight balancing during training

---

### 2. Overlapping Objects

Objects such as Fire Alarm and Safety Switch Panel frequently appeared close together.

**Solution**

- Tuned IoU thresholds
- Optimized Non-Maximum Suppression (NMS)

---

### 3. Deployment Constraints

The original model required more computational resources than desired.

**Solution**

- Exported the model to ONNX
- Reduced memory usage
- Improved inference speed

---

# Optimizations

Several optimizations were introduced to improve performance:

- Advanced data augmentation
- Learning rate scheduling
- Hyperparameter tuning
- ONNX model export
- Lightweight deployment optimization
- Fine-tuning with pretrained COCO weights

These improvements enhanced both detection accuracy and inference efficiency.

---

# Performance Evaluation

The trained model demonstrated strong performance across all evaluation metrics.

### Key Observations

- mAP@50 reached **87.8%**
- Precision exceeded **92%**
- Stable Recall of **83.4%**
- Reliable F1 Score of **87.5%**
- Near-perfect detection for Fire Extinguisher and First Aid Box
- Minor confusion only between visually similar gas tanks
- Successfully achieved **25 FPS** real-time inference

---

# Future Work

The project can be further enhanced through:

- Development of a Web Dashboard
- Multi-camera (360°) monitoring
- Larger datasets with synthetic image generation
- Real-time astronaut alert system
- Cloud deployment
- Edge AI optimization
- Expansion to industrial safety, hospitals, defense, and smart manufacturing

---

# Impact

AstroShield demonstrates how artificial intelligence can significantly improve astronaut safety by continuously monitoring life-saving equipment inside space stations.

The system enables:

- Faster emergency response
- Continuous automated monitoring
- Reduced human oversight
- Improved mission safety
- Reliable real-time equipment tracking

Although developed as a hackathon project, AstroShield has strong potential for future deployment in space exploration as well as industrial and high-risk environments.

---

# Tech Stack

- Python
- YOLOv8
- Ultralytics
- OpenCV
- NumPy
- ONNX
- CUDA (GPU Inference)

---

# Installation

```bash
git clone https://github.com/dsrisaigaensh/HACKWITHHYDRABAD.git

cd HACKWITHHYDRABAD

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

# Run Detection

```bash
python detect.py
```

or

```bash
yolo predict model=best.pt source=test_images/
```

---

# Results

The trained model successfully performs real-time safety equipment detection at **25 FPS**, making it suitable for intelligent monitoring applications in space stations and other safety-critical environments.

---
## Output Screenshots

### Real-Time Object Detection

<p align="center">
  <img src="images/detections.png" alt="Detection Results" width="900">
</p>

The model successfully detects mission-critical safety equipment including Oxygen Tanks, Nitrogen Tanks, Fire Extinguishers, First Aid Boxes, Fire Alarms, Emergency Phones, and Safety Switch Panels in real time.

---

## Training Performance

<p align="center">
  <img src="images/training_results.png" alt="Training Results" width="700">
</p>

**Final Model Performance**

| Metric | Score |
|--------|-------|
| Precision | **92.3%** |
| Recall | **83.4%** |
| mAP@50 | **87.8%** |
| mAP@50-95 | **78.9%** |
| Inference Speed | **25 FPS** |

---

## Sample Detection Output

<p align="center">
  <img src="images/sample_output.png" alt="Sample Detection" width="900">
</p>

The trained YOLOv8 model accurately detects multiple safety objects with high confidence, making it suitable for real-time astronaut safety monitoring.
---

## Team

**Amaravati Avengers**

**HackWithHyderabad Hackathon Project**
