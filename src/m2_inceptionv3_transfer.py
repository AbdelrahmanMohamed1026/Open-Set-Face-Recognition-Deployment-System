import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceRecognitionModel:
    """Handles the architecture, training, and evaluation of the InceptionV3 transfer learning model."""
    
    def __init__(self, processed_data_dir="data/processed", model_save_path="models/inceptionv3_face_recognition.h5"):
        self.processed_data_dir = Path(processed_data_dir)
        self.model_save_path = Path(model_save_path)
        self.input_shape = (299, 299, 3)
        self.batch_size = 32
        
        # Will be populated by the generators
        self.num_classes = None
        self.model = None

    def setup_data_generators(self):
        """Initializes the Keras generators pointing to the preprocessed splits."""
        logging.info("Setting up data generators...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            self.processed_data_dir / 'train',
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            self.processed_data_dir / 'val',
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            self.processed_data_dir / 'test',
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = train_gen.num_classes
        return train_gen, val_gen, test_gen

    def build_model(self):
        """Constructs the transfer learning architecture with a frozen base and custom head."""
        logging.info("Building InceptionV3 model architecture...")
        
        # 1. Load the base model without the classification head
        base_model = InceptionV3(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )
        
        # 2. Freeze the base model layers
        base_model.trainable = False
        
        # 3. Construct the custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # 4. Compile the final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        logging.info(f"Model built successfully with {self.num_classes} output classes.")
        self.model.summary(print_fn=logging.info)

    def train_model(self, train_gen, val_gen, epochs=20):
        """Executes the training loop with callbacks to monitor and optimize performance."""
        logging.info("Initializing training callbacks...")
        
        # Ensure models directory exists
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=3, 
                min_lr=1e-6,
                verbose=1
            ),
            # Saves the best model as it trains so you don't lose progress if it crashes
            ModelCheckpoint(
                filepath=str(self.model_save_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logging.info("Starting model training...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )
        
        return history

    def evaluate_model(self, test_gen):
        """Evaluates the final model against the unseen test set."""
        logging.info("Evaluating model on the test set...")
        loss, accuracy = self.model.evaluate(test_gen, verbose=1)
        logging.info(f"Test Set Evaluation -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, accuracy


if __name__ == "__main__":
    # Ensure TensorFlow utilizes the GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        logging.info(f"GPU acceleration enabled: {physical_devices}")
    else:
        logging.warning("No GPU found. Training will proceed on CPU and may take significantly longer.")

    pipeline = FaceRecognitionModel()
    
    # 1. Setup Data
    train_gen, val_gen, test_gen = pipeline.setup_data_generators()
    
    # 2. Build Architecture
    pipeline.build_model()
    
    # 3. Train
    history = pipeline.train_model(train_gen, val_gen, epochs=20)
    
    # 4. Evaluate & Save
    pipeline.evaluate_model(test_gen)
    logging.info(f"Phase 2 Complete. Model saved to {pipeline.model_save_path}")