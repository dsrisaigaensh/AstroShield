# SafetyDetectionYOLOv8

## Overview
SafetyDetectionYOLOv8 is a real-time safety equipment detection system built using the YOLOv8 object detection model. The project is designed to help workplaces automatically monitor safety compliance by detecting helmets, vests, masks, and other personal protective equipment (PPE) in images or video streams.

---

## Features
- Detects multiple safety equipment classes in real time.
- Works with images, videos, or live webcam streams.
- Fast and lightweight using the YOLOv8 architecture.
- Supports custom datasets for training new models.

---

## Methodology
1. **Data Preparation**: Images are labeled for different safety equipment classes using standard formats compatible with YOLO.
2. **Model Training**: YOLOv8 is trained on the dataset to recognize different safety equipment. The model uses transfer learning with pre-trained weights for faster convergence.
3. **Inference**: The trained model predicts safety equipment presence on images, videos, or webcam streams.
4. **Evaluation**: Model performance is measured using metrics like mAP50, precision, and recall to ensure reliability.

---

## Installation
```bash
git clone https://github.com/dsrisaigaensh/HACKWITHHYDRABAD.git
cd SafetyDetectionYOLOv8
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
