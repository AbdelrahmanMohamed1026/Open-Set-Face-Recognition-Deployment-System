# Phase 1: Face Detection and Dataset Preprocessing

## 1. Face Detection Strategy
- **Model:** OpenCV's deep neural network (DNN) module using the pre-trained ResNet-10 Single Shot MultiBox Detector (SSD).
- **Configuration:** Utilized `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`.
- **Detection Parameters:** Bounding boxes are filtered using a strict confidence threshold of **0.5**. The pipeline isolates the regions of interest (ROI) and inherently supports extracting multiple valid face crops from a single source frame.

## 2. Image Preprocessing & Normalization
- **Color Space Mapping:** All inputs are standardized to RGB format. Grayscale inputs are automatically converted to 3-channel RGB representation prior to processing.
- **Resolution:** Face crops are resized to `(299, 299)` using OpenCV. This strict dimensional standard ensures architectural compatibility with advanced transfer learning models (e.g., Xception, InceptionV3) expected in Phase 2.
- **Normalization:** Pixel matrices are rescaled by `1./255`. This translates the standard `[0, 255]` integer limits into `[0.0, 1.0]` floating-point tensors, which stabilizes gradient descent during future model training.

## 3. Dataset Splitting Policy
To ensure robust evaluation without data leakage, the pipeline enforces a reproducible stratified split (utilizing `random_state=42`):
- **Test Set:** 20% of the entire dataset.
- **Validation Set:** 20% of the remaining training portion (equating to 16% of the total dataset).
- **Training Set:** The remaining 80% of the training portion (equating to 64% of the total dataset).

## 4. Data Augmentation Policy
Spatial augmentations are strictly isolated to the training subset via Keras `ImageDataGenerator` to mitigate overfitting and improve generalization on unseen angles:
- **Rotation Range:** 20 degrees
- **Width/Height Shift:** 20% (0.2)
- **Horizontal Flip:** Enabled
- *Note: The Validation and Test sets utilize pixel normalization exclusively, guaranteeing unmodified evaluation benchmarks.*