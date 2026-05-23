# Phase 3: Optimization and Open-Set Recognition Report

## 1. Model Optimization & Quantization
To ensure the deployment application runs smoothly (even without a GPU), the trained Keras model was exported into two highly optimized formats:
- **ONNX (Open Neural Network Exchange):** The model was converted using `tf2onnx` with Opset 13. ONNX standardizes the computational graph, allowing the Streamlit application to utilize `onnxruntime` for fast, framework-agnostic inference.
- **TensorFlow Lite (TFLite):** The model was converted using `TFLiteConverter`. Default dynamic range quantization was applied, drastically reducing the model's file size and memory footprint while maintaining precision.

## 2. Embedding Extraction
Standard closed-set classification (Softmax) fails in real-world scenarios because it forces the model to choose one of the 105 known identities, even if a stranger is in the camera frame. 

To solve this, the final classification layer (`Dense 105`) and the Dropout layer were computationally stripped from the graph. The optimized ONNX and TFLite models now output from the penultimate `Dense(128)` layer. This transforms the model from a classifier into a feature extractor, outputting a 128-dimensional embedding vector representing the unique geometry of the inputted face.

## 3. Open-Set Recognition Logic
Face verification is now handled via **Cosine Similarity**, which measures the angular distance between two embedding vectors in a 128-dimensional space.
- **Reference Database:** Known individuals have their embeddings saved in a dictionary.
- **Similarity Threshold:** `0.5`. 
- **Inference Workflow:** An incoming face is passed through the embedding model. Its 128-D vector is compared against all vectors in the database. 
- **Non-Defined Class:** If the highest cosine similarity score fails to breach the `0.5` threshold, the system explicitly rejects the face and assigns it the class `"Non-Defined"`, successfully achieving open-set recognition capability.