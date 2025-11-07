"""
Step 4: Image Classification Model for Civic Issue Urgency
Fine-tuned CNN using ResNet50/EfficientNet with transfer learning
Classifies civic issue images into urgency levels (Low/Medium/High)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import ResNet50, EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ TensorFlow/Keras imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("💡 Install with: pip install tensorflow pillow")

class CivicImageClassifier:
    """
    CNN-based image classifier for civic issue urgency detection
    Uses transfer learning with ResNet50 or EfficientNet
    """
    
    def __init__(self, model_type='resnet50', input_shape=(224, 224, 3)):
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.class_names = ['LOW', 'MEDIUM', 'HIGH']
        self.num_classes = len(self.class_names)
        
        # Training configuration
        self.batch_size = 16
        self.epochs = 20
        self.learning_rate = 0.001
        self.fine_tune_epochs = 10
        self.fine_tune_lr = 0.0001
        
        print(f"🏗️ Initialized {model_type.upper()} Image Classifier")
        print(f"📐 Input shape: {input_shape}")
    
    def create_synthetic_image_data(self, samples_per_class=100):
        """
        Create synthetic image dataset with proper directory structure
        Since we don't have real images, we'll create colored placeholders
        """
        print("🎨 Creating Synthetic Image Dataset...")
        
        # Create directories
        data_dir = Path('../data/images')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Image categories and their characteristics
        image_categories = {
            'HIGH': {
                'description': 'High urgency: Exposed wires, deep potholes, major damage, sewage overflow',
                'color_scheme': [(139, 0, 0), (255, 0, 0), (128, 0, 0)],  # Dark red tones
                'patterns': ['lines', 'cracks', 'damage']
            },
            'MEDIUM': {
                'description': 'Medium urgency: Moderate potholes, broken lights, garbage, minor leaks',
                'color_scheme': [(255, 165, 0), (255, 140, 0), (255, 69, 0)],  # Orange tones
                'patterns': ['spots', 'moderate_damage', 'wear']
            },
            'LOW': {
                'description': 'Low urgency: Small cracks, minor litter, cosmetic issues, paint fading',
                'color_scheme': [(34, 139, 34), (0, 128, 0), (0, 100, 0)],  # Green tones
                'patterns': ['small_spots', 'minor_wear', 'cosmetic']
            }
        }
        
        image_data = []
        
        for urgency_level, config in image_categories.items():
            level_dir = data_dir / urgency_level
            level_dir.mkdir(exist_ok=True)
            
            print(f"📸 Creating {samples_per_class} images for {urgency_level} urgency...")
            
            for i in range(samples_per_class):
                # Create synthetic image with realistic civic issue patterns
                img = self._create_synthetic_civic_image(config, self.input_shape[:2])
                
                # Save image
                filename = f"{urgency_level.lower()}_{i:03d}.png"
                filepath = level_dir / filename
                
                # Convert to PIL and save
                from PIL import Image
                img_pil = Image.fromarray(img.astype('uint8'))
                img_pil.save(filepath)
                
                image_data.append({
                    'filepath': str(filepath),
                    'urgency_level': urgency_level,
                    'category': config['description'].split(':')[1].strip(),
                    'urgency_score': {'LOW': 3, 'MEDIUM': 6, 'HIGH': 9}[urgency_level]
                })
        
        # Save metadata
        metadata_df = pd.DataFrame(image_data)
        metadata_df.to_csv(data_dir / 'image_metadata.csv', index=False)
        
        print(f"✅ Created {len(image_data)} synthetic images")
        print(f"📊 Distribution: {metadata_df['urgency_level'].value_counts().to_dict()}")
        
        return metadata_df
    
    def _create_synthetic_civic_image(self, config, size):
        """Create a synthetic civic issue image with realistic patterns"""
        height, width = size
        
        # Base image with random texture
        np.random.seed(None)  # Different seed each time
        base_color = config['color_scheme'][np.random.randint(0, len(config['color_scheme']))]
        
        # Create base image
        img = np.ones((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = base_color[0] + np.random.randint(-30, 30, (height, width)).clip(0, 255)
        img[:, :, 1] = base_color[1] + np.random.randint(-30, 30, (height, width)).clip(0, 255)
        img[:, :, 2] = base_color[2] + np.random.randint(-30, 30, (height, width)).clip(0, 255)
        
        # Add pattern based on urgency level
        pattern = np.random.choice(config['patterns'])
        
        if pattern in ['lines', 'cracks']:
            # Add crack-like lines for high urgency
            for _ in range(np.random.randint(3, 8)):
                start_x, start_y = np.random.randint(0, width), np.random.randint(0, height)
                end_x, end_y = np.random.randint(0, width), np.random.randint(0, height)
                thickness = np.random.randint(2, 6)
                
                # Simple line drawing
                steps = max(abs(end_x - start_x), abs(end_y - start_y))
                if steps > 0:
                    for step in range(steps):
                        x = int(start_x + (end_x - start_x) * step / steps)
                        y = int(start_y + (end_y - start_y) * step / steps)
                        if 0 <= x < width and 0 <= y < height:
                            img[max(0, y-thickness):min(height, y+thickness), 
                                max(0, x-thickness):min(width, x+thickness)] = [0, 0, 0]
        
        elif pattern in ['spots', 'moderate_damage']:
            # Add spots for medium urgency
            for _ in range(np.random.randint(5, 15)):
                center_x, center_y = np.random.randint(20, width-20), np.random.randint(20, height-20)
                radius = np.random.randint(5, 15)
                
                # Create circular damage
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = [50, 50, 50]  # Dark spots
        
        # Add noise for realism
        noise = np.random.randint(-20, 20, img.shape)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def create_model(self, pretrained=True):
        """
        Create CNN model with transfer learning
        """
        print(f"🏗️ Creating {self.model_type.upper()} model...")
        
        # Base model selection
        if self.model_type == 'resnet50':
            base_model = ResNet50(
                weights='imagenet' if pretrained else None,
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet' if pretrained else None,
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu', name='dense_512'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', name='dense_256'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', name='dense_128'),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        self.base_model = base_model
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"🔒 Base model frozen: {len(base_model.layers)} layers")
        
        return model
    
    def create_data_generators(self, data_dir, validation_split=0.2, test_split=0.1):
        """
        Create data generators with augmentation
        """
        print("📊 Creating Data Generators...")
        
        # Data augmentation for training (as specified)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,          # Rotation
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,       # Flip
            brightness_range=[0.8, 1.2], # Brightness adjustment
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split + test_split
        )
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split + test_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        print(f"📈 Training samples: {train_generator.samples}")
        print(f"📊 Validation samples: {validation_generator.samples}")
        print(f"🏷️ Classes: {list(train_generator.class_indices.keys())}")
        
        return train_generator, validation_generator
    
    def train_initial(self, train_generator, validation_generator):
        """
        Initial training with frozen base model
        """
        print("\n🚀 Starting Initial Training (Frozen Base)...")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                monitor='val_loss'
            ),
            callbacks.ModelCheckpoint(
                '../models/civic_image_classifier_initial.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Calculate class weights for imbalanced data
        class_weights = self._calculate_class_weights(train_generator)
        print(f"📊 Class weights: {class_weights}")
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        self.history = history
        print("✅ Initial training completed!")
        
        return history
    
    def fine_tune_model(self, train_generator, validation_generator):
        """
        Fine-tune model by unfreezing some layers
        """
        print("\n🔧 Starting Fine-tuning (Unfreezing Layers)...")
        
        # Unfreeze top layers of base model
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.base_model.layers) - 20
        
        # Freeze all layers except the last few
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"🔓 Unfrozen last {len(self.base_model.layers) - fine_tune_at} layers")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Fine-tune training
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            callbacks.ModelCheckpoint(
                '../models/civic_image_classifier_finetuned.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=self.fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combine histories
        if self.history:
            for key in fine_tune_history.history.keys():
                self.history.history[key].extend(fine_tune_history.history[key])
        
        print("✅ Fine-tuning completed!")
        return fine_tune_history
    
    def _calculate_class_weights(self, generator):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get class counts
        class_counts = {}
        for class_name, class_idx in generator.class_indices.items():
            class_counts[class_idx] = 0
        
        # Count samples per class
        for i in range(len(generator)):
            batch_x, batch_y = generator[i]
            class_indices = np.argmax(batch_y, axis=1)
            for idx in class_indices:
                class_counts[idx] += 1
        
        # Calculate weights
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        weights = compute_class_weight(
            'balanced',
            classes=np.array(classes),
            y=np.repeat(classes, counts)
        )
        
        return dict(zip(classes, weights))
    
    def evaluate_model(self, validation_generator):
        """
        Comprehensive model evaluation
        """
        print("\n📊 Evaluating Model...")
        
        # Predictions
        validation_generator.reset()
        predictions = self.model.predict(validation_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # True labels
        true_classes = validation_generator.classes
        
        # Class names
        class_labels = list(validation_generator.class_indices.keys())
        
        # Accuracy
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"🎯 Overall Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_labels': class_labels
        }
    
    def create_visualizations(self, evaluation_results):
        """Create comprehensive visualizations"""
        print("\n📈 Creating Visualizations...")
        
        viz_dir = Path('../visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Training History
        if self.history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy plot
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Loss plot
            ax2.plot(self.history.history['loss'], label='Training Loss')
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Confusion Matrix
        cm = evaluation_results['confusion_matrix']
        class_labels = evaluation_results['class_labels']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - Image Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'image_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Per-class Performance
        report = evaluation_results['classification_report']
        metrics_df = pd.DataFrame(report).transpose().iloc[:-3]  # Exclude avg rows
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_labels))
        width = 0.25
        
        plt.bar(x - width, metrics_df['precision'], width, label='Precision', alpha=0.8)
        plt.bar(x, metrics_df['recall'], width, label='Recall', alpha=0.8)
        plt.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, class_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to ../visualizations/")
    
    def predict_image(self, image_path):
        """
        Predict urgency for a single image
        """
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=self.input_shape[:2])
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            confidence = np.max(predictions[0])
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Convert to urgency score
            urgency_scores = {'LOW': 3, 'MEDIUM': 6, 'HIGH': 9}
            urgency_score = urgency_scores[predicted_class]
            
            return {
                'urgency_level': predicted_class,
                'urgency_score': urgency_score,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            }
            
        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None
    
    def save_model(self, filepath='../models/civic_image_classifier.h5'):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save!")
            return
        
        # Create models directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'training_params': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'fine_tune_epochs': self.fine_tune_epochs,
                'fine_tune_lr': self.fine_tune_lr
            }
        }
        
        metadata_path = str(filepath).replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {filepath}")
        print(f"📋 Metadata saved to: {metadata_path}")
    
    def load_model(self, filepath='../models/civic_image_classifier.h5'):
        """Load a saved model"""
        try:
            self.model = keras.models.load_model(filepath)
            
            # Load metadata
            metadata_path = str(filepath).replace('.h5', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_type = metadata['model_type']
                self.input_shape = tuple(metadata['input_shape'])
                self.class_names = metadata['class_names']
                self.num_classes = metadata['num_classes']
            
            print(f"✅ Model loaded from: {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")

def main():
    """Main training pipeline"""
    print("🎨 CIVIC ISSUE IMAGE CLASSIFICATION - STEP 4")
    print("=" * 60)
    
    try:
        # Initialize classifier
        classifier = CivicImageClassifier(model_type='resnet50')
        
        # Create synthetic dataset
        print("\n📸 Phase 1: Dataset Creation")
        metadata_df = classifier.create_synthetic_image_data(samples_per_class=50)  # Small for demo
        
        # Create model
        print("\n🏗️ Phase 2: Model Architecture")
        model = classifier.create_model(pretrained=True)
        model.summary()
        
        # Create data generators
        print("\n📊 Phase 3: Data Preparation")
        train_gen, val_gen = classifier.create_data_generators('../data/images')
        
        # Initial training
        print("\n🚀 Phase 4: Initial Training")
        classifier.train_initial(train_gen, val_gen)
        
        # Fine-tuning
        print("\n🔧 Phase 5: Fine-tuning")
        classifier.fine_tune_model(train_gen, val_gen)
        
        # Evaluation
        print("\n📊 Phase 6: Evaluation")
        results = classifier.evaluate_model(val_gen)
        
        # Visualizations
        classifier.create_visualizations(results)
        
        # Save model
        classifier.save_model()
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"🎯 Final Accuracy: {results['accuracy']:.4f}")
        print(f"📁 Model saved to: ../models/civic_image_classifier.h5")
        
        # Demo prediction
        print(f"\n🔮 Testing Sample Predictions...")
        sample_images = list(Path('../data/images').rglob('*.png'))[:3]
        for img_path in sample_images:
            result = classifier.predict_image(img_path)
            if result:
                print(f"📸 {img_path.name}: {result['urgency_level']} ({result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Quick Fix Options:")
        print(f"1. Install TensorFlow: pip install tensorflow pillow")
        print(f"2. Use CPU-only version for testing")
        print(f"3. Run on Google Colab with GPU")

if __name__ == "__main__":
    main()