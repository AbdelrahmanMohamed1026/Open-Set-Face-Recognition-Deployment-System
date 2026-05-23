import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionPipeline:
    def __init__(self, prototxt_path="assets/deploy.prototxt", 
                 model_path="assets/res10_300x300_ssd_iter_140000.caffemodel",
                 onnx_path="models/face_recognition.onnx",
                 conf_threshold=0.5, sim_threshold=0.5):
        
        self.conf_threshold = conf_threshold
        self.sim_threshold = sim_threshold
        
        # Initialize OpenCV DNN Face Detector
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        # Initialize ONNX Runtime Session for fast feature extraction
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def detect_faces(self, image):
        """Detects faces in an RGB image and returns bounding boxes and cropped arrays."""
        h, w = image.shape[:2]
        # Convert RGB to BGR for the OpenCV detector
        blob_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(blob_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Boundary checks
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                if startX < endX and startY < endY:
                    face_crop = image[startY:endY, startX:endX]
                    faces.append({
                        "box": (startX, startY, endX, endY), 
                        "crop": face_crop,
                        "confidence": confidence
                    })
        return faces

    def get_embedding(self, face_crop):
        """Processes a cropped face and extracts the 128-D ONNX embedding."""
        face_resized = cv2.resize(face_crop, (299, 299))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        embedding = self.ort_session.run(None, {self.input_name: face_expanded})[0]
        return embedding

    def recognize(self, embedding, reference_db):
        """Compares a 128-D embedding against a dictionary of known embeddings."""
        if not reference_db:
            return "Non-Defined", 0.0
            
        best_match = "Non-Defined"
        highest_sim = -1.0
        
        for name, ref_emb in reference_db.items():
            sim = cosine_similarity(embedding, ref_emb)[0][0]
            if sim > highest_sim:
                highest_sim = sim
                if sim >= self.sim_threshold:
                    best_match = name
                    
        return best_match, highest_sim
        
    def process_frame(self, image, reference_db):
        """Full pipeline: detects faces, draws boxes/labels, and returns the annotated image."""
        annotated_image = image.copy()
        faces = self.detect_faces(image)
        
        for face in faces:
            startX, startY, endX, endY = face["box"]
            embedding = self.get_embedding(face["crop"])
            identity, score = self.recognize(embedding, reference_db)
            
            # Choose color based on recognition (Green for known, Red for unknown)
            color = (0, 255, 0) if identity != "Non-Defined" else (255, 0, 0)
            
            # Draw Box
            cv2.rectangle(annotated_image, (startX, startY), (endX, endY), color, 2)
            
            # Draw Label
            text = f"{identity} ({score:.2f})"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(annotated_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return annotated_image