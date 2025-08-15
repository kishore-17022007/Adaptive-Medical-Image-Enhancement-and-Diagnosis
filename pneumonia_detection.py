# PNEUMONIA DETECTION FROM CHEST X-RAYS - IMPROVED WORKING CODE
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import os

# 1. SETUP ======================================================
IMG_SIZE = (224, 224)  # ResNet input size
BATCH_SIZE = 32
EPOCHS = 15  # Increased epochs to allow for early stopping
LR = 0.0001

# 2. DATA PIPELINE ==============================================
base_dir = 'chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators with balanced classes
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42  # Fixed seed for reproducibility
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")  # Debug output

# 3. IMPROVED MODEL BUILDING ====================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling='avg'  # Directly add global pooling
)

base_model.trainable = False

# Simplified model architecture
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    base_model,
    layers.Dropout(0.3),  # Reduced dropout
    layers.Dense(128, activation='relu'),  # Smaller dense layer
    layers.Dense(1, activation='sigmoid')
])

# Mixed precision training for better performance
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model.compile(
    optimizer=optimizers.Adam(LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp')
    ]
)

# 4. ENHANCED TRAINING ==========================================
callbacks = [
    callbacks.EarlyStopping(
        monitor='val_auc',  # Changed to monitor AUC
        patience=5,
        mode='max',
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Added class weights
    verbose=2
)

# 5. COMPREHENSIVE EVALUATION ===================================
def plot_history(history):
    plt.figure(figsize=(18, 6))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # AUC
    plt.subplot(1, 3, 2)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Load best model for final evaluation
model = models.load_model('best_model.h5')

# Test evaluation
test_results = model.evaluate(test_generator)
print("\nFINAL TEST METRICS:")
print(f"Accuracy: {test_results[1]:.4f}")
print(f"Precision: {test_results[2]:.4f}")
print(f"Recall: {test_results[3]:.4f}")
print(f"AUC: {test_results[4]:.4f}")
print(f"True Positives: {test_results[5]}")
print(f"False Positives: {test_results[6]}")

# Enhanced confusion matrix
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator) > 0.5

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'],
            annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.show()

# Detailed classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# 6. MODEL SAVING ===============================================
model.save('pneumonia_detection.h5')
print("Model saved as pneumonia_detection.h5")

# Save the class indices for deployment
import json
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Class indices saved to class_indices.json")