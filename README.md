# 🦺 Jacket and Helmet Detection System

## 📌 Project Overview

This project implements an AI-based object detection system to monitor **safety compliance** in industrial and construction zones. It uses a **custom-trained YOLOv11 model** to detect and classify individuals wearing:
- Only helmets
- Only safety jackets
- Both helmet and jacket
- Neither

The system leverages **Kubeflow**, **Roboflow**, and **Ultralytics YOLOv11** to train and deploy the solution at scale.

---

## 🎯 Objectives

- Detect people in images/videos using real-time object detection.
- Classify safety gear compliance into four distinct categories.
- Annotate datasets with high accuracy using Roboflow.
- Train and evaluate a YOLOv11 model on a custom dataset.
- Perform inference on test images and videos.

---

## 🛠️ Tech Stack & Tools

| Tool          | Purpose                              |
|---------------|---------------------------------------|
| **Ubuntu OS** | Robust AI development environment     |
| **Docker**    | Containerization of development stack |
| **Kubeflow**  | End-to-end ML workflow orchestration  |
| **YOLOv11**   | Object detection framework            |
| **Roboflow**  | Image annotation & preprocessing      |
| **PyTorch**   | Deep learning framework               |
| **Ultralytics** | YOLO training & inference interface |
| **Python**    | Scripting and automation              |

---

## 🧾 Annotation Process

1. Annotate images/videos using **Roboflow**.
2. Assign labels:
   - `helmet only`
   - `jacket only`
   - `both`
   - `none`
3. Export in **YOLOv11 format**.
4. Resize images to **640x640** during export.

---

## ⚙️ Setup & Training Pipeline

### 1️⃣ Access Kubeflow
Open Firefox and visit:
```
http://<KUBEFLOW_IP>:<PORT>
```
Login with your credentials.

### 2️⃣ Set Up Notebook Server
- Image: `PyTorch Image`
- CPU: `8 cores`
- RAM: `16 GB`
- GPU: `10/20 GB`
- Storage: None

### 3️⃣ Install Dependencies
```bash
cd /workspace/
pip install ultralytics
apt update
apt install ffmpeg libsm6 libxext6 -y
```

### 4️⃣ Test Inference
```bash
yolo task=detect mode=predict model=yolo11n.pt source="testimg.jpg"
```
Fix NumPy version error (if any):
```bash
pip install numpy==1.23.4
```

Run on CPU:
```bash
yolo task=detect mode=predict model=yolo11n.pt source="testimg.jpg" device=cpu
```

---

## 📦 Dataset Preparation

### 📥 Download Dataset from Roboflow
```bash
mkdir custom_dataset
cd custom_dataset/
curl -L "<roboflow_download_url>" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip
cd ..
```

### 🛠️ Update `data.yaml`
Edit `custom_dataset/data.yaml`:
```yaml
train: /workspace/custom_dataset/train/images
val: /workspace/custom_dataset/valid/images
test: /workspace/custom_dataset/test/images
```
Rename it to:
```bash
mv data.yaml custom_data.yaml
```

---

## 🧠 Model Training

```bash
yolo task=detect mode=train model=yolo11n.pt data="custom_dataset/custom_data.yaml" epochs=10 imgsz=640
```

If GPU is not detected:
```bash
nvcc --version
# Check CUDA version and reinstall matching PyTorch build
pip uninstall torch torchvision torchaudio
pip install torch==<version> torchvision==<version> torchaudio==<version> -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 📈 Evaluation & Inference

- Results: `runs/detect/train/`
- Copy best model:
```bash
cp runs/detect/train/weights/best.pt /workspace
```

### 🔍 Run Final Inference
```bash
yolo task=detect mode=predict model=best.pt source="vid.mp4"
```

- Output: `runs/detect/predict/`

---

## ✅ Conclusion

This project demonstrates the successful use of deep learning for **automated safety gear detection**. The custom YOLOv11 model performed well in real-world test scenarios.

---

## 📚 Learnings

- In-depth knowledge of YOLO-based object detection.
- Efficient annotation using Roboflow.
- Seamless deployment using Kubeflow notebooks.
- Real-time inference with GPU and CPU support.
- Importance of structured dataset labeling in safety-critical applications.

---

## 💡 Future Enhancements

- Integrate with CCTV systems for live monitoring.
- Deploy model as a REST API or edge device.
- Add alert system for non-compliant detections.

---
