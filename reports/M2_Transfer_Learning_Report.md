# Phase 2: Transfer Learning Architecture & Training Report

## 1. Model Architecture Setup
The face recognition model utilizes a **Transfer Learning** paradigm to leverage deep feature representations trained on ImageNet.
- **Base Backbone:** `InceptionV3` (weights initialized to `imagenet`, top layers excluded).
- **Input Dimensions:** `(299, 299, 3)` to natively match InceptionV3's optimal tensor shape.
- **Base State:** Frozen. Training is strictly limited to the newly added classification head.

**Custom Classification Head:**
1. **Global Average Pooling 2D:** Flattens the final `(8, 8, 2048)` feature map from the base model into a dense `(2048,)` vector, aggressively reducing parameter count to prevent overfitting.
2. **Dense Layer:** 128 units with `ReLU` activation for non-linear feature combination.
3. **Dropout Layer:** Set to `0.5` (50%) to enforce regularization during training.
4. **Output Dense Layer:** 105 units matching the dataset's identities, using `softmax` activation to output a categorical probability distribution.

## 2. Training Configuration
- **Optimizer:** Adam
- **Initial Learning Rate:** `0.001`
- **Loss Function:** Categorical Crossentropy (required for multi-class, one-hot encoded targets).
- **Primary Metric:** Accuracy

## 3. Dynamic Optimization Strategies
Two primary callbacks were implemented to monitor validation performance epoch-by-epoch:
- **EarlyStopping:** Monitors `val_loss`. Halts training if the model fails to improve for 5 consecutive epochs, restoring the weights from the epoch with the lowest validation loss.
- **ReduceLROnPlateau:** Monitors `val_loss`. Automatically decays the learning rate by a factor of 0.2 if the loss plateaus for 3 epochs, allowing the optimizer to settle into finer local minima.

## 4. Evaluation Results
*(Run `src/m2_inceptionv3_transfer.py` and log the output numbers here)*
- **Test Loss:** `[INSERT TEST LOSS HERE]`
- **Test Accuracy:** `[INSERT TEST ACCURACY HERE]`