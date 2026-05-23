# Phase 4: Live Deployment and Interface Report

## 1. Application Architecture
The final face recognition system employs a decoupled architecture:
- **Backend Inference (`src/m4_live_inference.py`):** Handles all heavy OpenCV matrix operations and ONNX runtime execution. By abstracting this logic, the UI remains responsive and framework-agnostic.
- **Frontend UI (`app/streamlit_app.py`):** Built with Streamlit to handle live webcam capturing, file uploads, and session-state memory for the similarity database.

## 2. Real-Time Inference Workflow
1. **Input:** An RGB frame is captured via file upload or Streamlit's `camera_input` component.
2. **Detection:** OpenCV's DNN SSD model detects faces, generating bounding boxes (`conf > 0.5`).
3. **Extraction:** Cropped ROIs are resized to `299x299` and normalized. The ONNX InceptionV3 model extracts the 128-D embedding.
4. **Comparison:** The embedding is compared against the `reference_db` maintained in Streamlit's session state.
5. **Visualization:** OpenCV draws bounding boxes and overlays the recognized identity (or "Non-Defined" if similarity is `< 0.5`) before passing the annotated array back to Streamlit for rendering.

## 3. Deployment Considerations
The application utilizes **ONNX Runtime** instead of raw TensorFlow. ONNX drastically reduces dependency sizes and cold-start times, making the Streamlit app highly portable and optimized for CPU-bound web server deployments.