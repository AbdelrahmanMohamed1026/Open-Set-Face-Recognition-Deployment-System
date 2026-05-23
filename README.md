# Open-Set Face Recognition & Deployment System

> **Acknowledgment:** This capstone project was developed as part of the **sprintsxACC — AI and Machine Learning** program.

---

## 📌 Overview

This repository contains a complete, end-to-end face recognition and detection pipeline. The project transitions a closed-set deep learning classifier into an **open-set, embedding-based verification system** capable of recognizing registered identities while explicitly rejecting unknown individuals by flagging them as `"Non-Defined"` — mirroring real-world security applications.

---

## ✨ Key Features

- **Robust Face Detection:** Utilizes OpenCV's Deep Neural Network (DNN) module with a pre-trained ResNet-10 Single Shot Detector (SSD) to accurately isolate bounding boxes in both static images and live video streams.
- **Transfer Learning:** Employs a pre-trained `InceptionV3` backbone to extract deep spatial feature maps, optimizing training efficiency and preventing catastrophic forgetting.
- **Open-Set Verification:** Strips the classification head to convert the network into a feature extractor, outputting 128-dimensional embedding vectors and relying on Cosine Similarity (threshold: `0.5`) to match faces against a dynamic database.
- **High-Performance Inference:** Converts the Keras model into **ONNX** and quantized **TensorFlow Lite (TFLite)** formats for high-speed, CPU-friendly inference in real-time applications.
- **Interactive Dashboard:** A decoupled **Streamlit** web application for registering new profiles and testing real-time detection via webcam or image uploads.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Core Machine Learning** | TensorFlow 2.x, Keras, Scikit-learn |
| **Computer Vision** | OpenCV (DNN module) |
| **Model Optimization** | ONNX Runtime, tf2onnx, TensorFlow Lite |
| **Web Deployment** | Streamlit |
| **Data Processing** | NumPy |
| **Dataset** | Kaggle `pins-face-recognition` (105 identities, ~17,500 images) |

---

## 🗂️ Repository Structure

```
face-recognition-capstone/
├── app/
│   └── streamlit_app.py                          # Interactive web interface
├── assets/
│   ├── deploy.prototxt                           # Face detector architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detector weights
├── models/
│   ├── face_recognition.onnx                     # Optimized ONNX model for deployment
│   ├── face_recognition.tflite                   # Quantized model for edge devices
│   └── inceptionv3_face_recognition.h5           # Original trained Keras model
├── reports/
│   ├── M1_Preprocessing_and_Detection.md
│   ├── M2_Transfer_Learning_Report.md
│   ├── M3_Optimization_and_OpenSet.md
│   └── M4_Deployment_Report.md
└── src/
    ├── m1_face_detection_preprocessing.py
    ├── m2_inceptionv3_transfer.py
    ├── m3_open_set_recognition.py
    └── m4_live_inference.py
```

---

## 🚀 Project Architecture & Phases

### Phase 1: Preprocessing & Detection (`m1_face_detection_preprocessing.py`)

- Downloads pre-trained Caffe models for OpenCV face detection.
- Extracts faces from the Kaggle dataset, converts them to RGB, and resizes to `(299, 299)`.
- Enforces a reproducible `64/16/20` stratified split (Train / Validation / Test).
- Generates Keras `ImageDataGenerators` with spatial augmentations (rotations, shifts, flips) applied exclusively to the training set.

### Phase 2: Transfer Learning (`m2_inceptionv3_transfer.py`)

- Instantiates `InceptionV3` with pre-trained ImageNet weights and freezes the base architecture.
- Attaches a custom classification head: `Global Average Pooling → Dense(128) → Dropout(0.5) → Softmax`.
- Trains using the Adam optimizer with categorical crossentropy, dynamic learning rate decay (`ReduceLROnPlateau`), and `EarlyStopping` to guard against overfitting.

### Phase 3: Open-Set Optimization (`m3_open_set_recognition.py`)

- Strips the final Softmax layer to convert the model into a 128-D feature extractor.
- Implements face verification using Cosine Similarity for threshold-based identity matching.
- Exports the finalized model to `.onnx` and `.tflite` formats for lightweight, deployment-ready inference.

### Phase 4: Live Inference Application (`m4_live_inference.py` & `streamlit_app.py`)

- Separates backend matrix operations (OpenCV + ONNX inference) from the Streamlit frontend.
- Allows users to dynamically build a session-state database of known faces.
- Processes live camera feeds to draw bounding boxes and overlay real-time similarity scores.

---

## ⚙️ Setup & Installation

**1. Clone the repository:**

```bash
git clone <your-repository-url>
cd face-recognition-capstone
```

**2. Install required dependencies:**

```bash
pip install tensorflow opencv-python numpy scikit-learn tf2onnx onnxruntime streamlit tqdm
```

**3. Prepare the data** *(only required if reproducing training):*

Download the `pins-face-recognition` dataset from Kaggle, then place the unzipped `105_classes_pins_dataset` folder at `data/raw/` in the project root.

---

## 💻 Usage

**Run the interactive Streamlit dashboard:**

```bash
streamlit run app/streamlit_app.py
```

> **Note:** If running on Google Colab, use `localtunnel` to expose the Streamlit port to a public URL.

**Reproduce the full backend pipeline** by executing the source scripts sequentially:

```bash
python src/m1_face_detection_preprocessing.py
python src/m2_inceptionv3_transfer.py
python src/m3_open_set_recognition.py
python src/m4_live_inference.py
```

---

## 📊 Future Improvements

- [ ] Add support for multi-face tracking in video streams
- [ ] Fine-tune the Cosine Similarity threshold per deployment environment
- [ ] Integrate a persistent face database (SQLite / Firebase) to replace session-state storage
- [ ] Containerize the application with Docker for reproducible deployment
- [ ] Explore ArcFace or FaceNet embeddings for improved verification accuracy
