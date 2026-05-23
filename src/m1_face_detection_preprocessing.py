import argparse
import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetector:
    """Handles loading the OpenCV DNN model and extracting face crops."""
    
    def __init__(self, models_dir="models", conf_thresh=0.5):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.conf_thresh = conf_thresh
        
        self.prototxt_path = self.models_dir / "deploy.prototxt"
        self.model_path = self.models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        self._ensure_models_exist()
        self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.model_path))

    def _ensure_models_exist(self):
        """Downloads the Caffe model files if they are not found locally."""
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not self.prototxt_path.exists():
            logging.info("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(proto_url, self.prototxt_path)
            
        if not self.model_path.exists():
            logging.info("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
            urllib.request.urlretrieve(model_url, self.model_path)

    def detect_and_crop(self, image_path, target_size=(299, 299)):
        """Detects faces in an image and returns a list of cropped, resized RGB faces."""
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        # Convert grayscale to RGB, or BGR to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        (h, w) = img.shape[:2]
        # Standard mean subtraction for the ResNet SSD model
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_thresh:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure boundaries are within the frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                if startX >= endX or startY >= endY:
                    continue

                face_crop = img[startY:endY, startX:endX]
                try:
                    face_crop = cv2.resize(face_crop, target_size)
                    faces.append(face_crop)
                except Exception as e:
                    logging.warning(f"Resize failed for {image_path}: {e}")

        return faces

class FaceRecognitionPipeline:
    """Manages dataset splitting, face extraction mapping, and generator creation."""
    
    def __init__(self, raw_data_dir, processed_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.detector = FaceDetector(conf_thresh=0.5)

    def _get_files_and_labels(self):
        """Maps out the dataset by identity."""
        file_paths = []
        labels = []
        
        for class_dir in self.raw_data_dir.iterdir():
            if class_dir.is_dir():
                identity = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    file_paths.append(str(img_path))
                    labels.append(identity)
                    
        return file_paths, labels

    def split_and_extract(self):
        """Splits the raw paths and processes faces into structural directories."""
        logging.info("Mapping raw dataset...")
        file_paths, labels = self._get_files_and_labels()
        
        # Policy: 80% train_val pool, 20% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Policy: 80% train, 20% val (from the 80% train_val pool)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        splits = {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
        
        for split_name, paths in splits.items():
            logging.info(f"Processing {split_name} split ({len(paths)} source images)...")
            split_dir = self.processed_dir / split_name
            
            extracted_count = 0
            for i, path_str in enumerate(paths):
                path = Path(path_str)
                identity = path.parent.name
                
                out_dir = split_dir / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                
                faces = self.detector.detect_and_crop(path)
                for j, face in enumerate(faces):
                    out_filename = f"{path.stem}_face_{j}.jpg"
                    # Convert back to BGR for OpenCV saving
                    cv2.imwrite(str(out_dir / out_filename), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    extracted_count += 1
                    
                if (i + 1) % 1000 == 0:
                    logging.info(f"  Processed {i+1}/{len(paths)} images in {split_name}...")
                    
            logging.info(f"Completed {split_name}: Extracted {extracted_count} faces.")

    def get_data_generators(self, batch_size=32):
        """Creates the Keras DataGenerators matching the augmented requirements."""
        logging.info("Building Keras Data Generators...")
        
        # Training includes augmentation and normalizaton
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        # Val and Test strictly use normalization only
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            self.processed_dir / 'train',
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            self.processed_dir / 'val',
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            self.processed_dir / 'test',
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Face Detection Pipeline")
    parser.add_argument("--raw_dir", type=str, default="data/raw/105_classes_pins_dataset", 
                        help="Path to the raw Kaggle dataset")
    args = parser.parse_args()

    pipeline = FaceRecognitionPipeline(raw_data_dir=args.raw_dir)
    
    # 1. Process images and save structures
    pipeline.split_and_extract()
    
    # 2. Verify generator instantiation 
    train_gen, val_gen, test_gen = pipeline.get_data_generators()
    logging.info("Phase 1 Pipeline Execution Complete.")