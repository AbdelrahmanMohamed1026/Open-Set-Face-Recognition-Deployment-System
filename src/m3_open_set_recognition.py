import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenSetOptimizer:
    def __init__(self, model_path="models/inceptionv3_face_recognition.h5", output_dir="models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_model = None
        self.embedding_model = None

    def load_and_prepare_embedding_model(self):
        """Loads the trained H5 model and strips the final classification head to output embeddings."""
        logging.info(f"Loading trained model from {self.model_path}...")
        self.base_model = tf.keras.models.load_model(self.model_path, compile=False)
        
        # We extract the output of the Dense(128) layer (which is before Dropout and the final Softmax)
        # In our M2 architecture, this is usually model.layers[-3]
        # To be safe, we search backwards for the first Dense layer before the final one
        dense_layers = [layer for layer in self.base_model.layers if isinstance(layer, tf.keras.layers.Dense)]
        if len(dense_layers) >= 2:
            embedding_layer = dense_layers[-2].output 
        else:
            embedding_layer = self.base_model.layers[-2].output
            
        self.embedding_model = tf.keras.Model(inputs=self.base_model.input, outputs=embedding_layer)
        logging.info(f"Embedding model prepared. Output shape: {self.embedding_model.output_shape}")

    def export_to_tflite(self):
        """Converts and applies default quantization for TensorFlow Lite deployment."""
        logging.info("Converting model to TFLite format with quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.embedding_model)
        
        # Apply default optimizations (dynamic range quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = self.output_dir / "face_recognition.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        logging.info(f"TFLite model saved to {tflite_path}")

    def export_to_onnx(self):
        """Converts the embedding model to ONNX format for cross-platform inference."""
        logging.info("Converting model to ONNX format...")
        onnx_path = self.output_dir / "face_recognition.onnx"
        
        # Define input signature matching InceptionV3
        spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
        
        # Convert to ONNX using opset 13
        model_proto, _ = tf2onnx.convert.from_keras(
            self.embedding_model, 
            input_signature=spec, 
            opset=13,
            output_path=str(onnx_path)
        )
        logging.info(f"ONNX model saved to {onnx_path}")


class OpenSetRecognizer:
    def __init__(self, similarity_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.reference_database = {} # Dictionary mapping 'Name' -> Embedding Vector

    def add_reference_identity(self, name, embedding):
        """Adds a known person's embedding to the database."""
        # Ensure embedding is 2D for cosine_similarity
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        self.reference_database[name] = embedding

    def recognize(self, test_embedding):
        """Compares a test embedding against all references using Cosine Similarity."""
        if not self.reference_database:
            return "Non-Defined", 0.0

        if len(test_embedding.shape) == 1:
            test_embedding = test_embedding.reshape(1, -1)

        best_match = "Non-Defined"
        highest_similarity = -1.0

        for name, ref_embedding in self.reference_database.items():
            # Calculate Cosine Similarity (returns 1.0 for identical, -1.0 for completely opposite)
            sim = cosine_similarity(test_embedding, ref_embedding)[0][0]
            
            if sim > highest_similarity:
                highest_similarity = sim
                if sim >= self.similarity_threshold:
                    best_match = name

        return best_match, highest_similarity

if __name__ == "__main__":
    # 1. Optimize and Export Models
    optimizer = OpenSetOptimizer()
    optimizer.load_and_prepare_embedding_model()
    optimizer.export_to_tflite()
    optimizer.export_to_onnx()
    
    # 2. Test Open-Set Logic
    logging.info("Testing Open-Set Cosine Similarity logic...")
    recognizer = OpenSetRecognizer(similarity_threshold=0.5)
    
    # Create dummy embeddings (128 dimensions) to verify the logic works
    dummy_known_face = np.random.rand(1, 128)
    dummy_unknown_face = np.random.rand(1, 128)
    
    # Register the known face
    recognizer.add_reference_identity("Test_Subject_A", dummy_known_face)
    
    # Test 1: Should match
    identity, score = recognizer.recognize(dummy_known_face)
    logging.info(f"Test 1 (Identical Face) -> Prediction: {identity}, Score: {score:.4f}")
    
    # Test 2: Should be Non-Defined
    identity, score = recognizer.recognize(dummy_unknown_face)
    logging.info(f"Test 2 (Random Face) -> Prediction: {identity}, Score: {score:.4f}")
    
    logging.info("Phase 3 Execution Complete.")