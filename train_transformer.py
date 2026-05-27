import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.training.transformer_model import build_transformer_model
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Configure GPU/MPS if available
if tf.config.list_physical_devices('GPU'):
    print("GPU detected!")

def normalize_landmarks(X):
    """
    Normalizes landmarks by centering each hand at its own wrist 
    and scaling independently.
    
    Each hand (21 points) is normalized separately so that:
    - Zero-padded hands stay as zeros (no broken centering)
    - Each hand is always wrist-centered and unit-scaled
    - Consistent regardless of whether 1 or 2 hands are present
    
    X shape: (N, 42, 3) for 2-hand data
    """
    normalized_X = []
    for landmarks in X:
        left = landmarks[:21].copy()
        right = landmarks[21:].copy()

        # Normalize LEFT hand
        if not np.all(left == 0):
            wrist = left[0]
            left = left - wrist
            dist = np.linalg.norm(left, axis=1)
            max_dist = np.max(dist) + 1e-6
            left = left / max_dist

        # Normalize RIGHT hand
        if not np.all(right == 0):
            wrist = right[0]
            right = right - wrist
            dist = np.linalg.norm(right, axis=1)
            max_dist = np.max(dist) + 1e-6
            right = right / max_dist

        normalized = np.vstack([left, right])
        normalized_X.append(normalized)
    return np.array(normalized_X)

class PerformanceLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('sparse_categorical_accuracy')
        val_acc = logs.get('val_sparse_categorical_accuracy')
        print(f"\n" + "="*40, flush=True)
        print(f"EPOCH {epoch+1} SUMMARY", flush=True)
        print(f"  Training Accuracy:   {train_acc:.4f}", flush=True)
        print(f"  Validation Accuracy: {val_acc:.4f}", flush=True)
        print("="*40 + "\n", flush=True)

def train_transformer():
    print("="*60)
    print("LANDMARK-BASED TRANSFORMER TRAINING")
    print("="*60)
    
    # 1. Load Data
    data_path = "dataset/landmarks.npz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run extraction first.")
        return
        
    print(f"Loading landmarks from {data_path}...")
    loaded = np.load(data_path)
    X = loaded['X']
    y = loaded['y']
    
    # Auto-detect format: 63 features = 1 hand (21x3), 126 features = 2 hands (42x3)
    num_features = X.shape[1]
    if num_features == 126:
        num_points = 42  # Two hands
        print(f"Detected 2-hand format: {num_features} features -> ({num_points}, 3)")
    elif num_features == 63:
        num_points = 21  # One hand
        print(f"Detected 1-hand format: {num_features} features -> ({num_points}, 3)")
    else:
        print(f"ERROR: Unexpected feature count: {num_features}. Expected 63 or 126.")
        return
    
    X = X.reshape(-1, num_points, 3)
    print(f"Total samples: {len(X)}, Input shape per sample: ({num_points}, 3)")
    
    # 2. Preprocess
    print("Normalizing landmarks (Wrist-centering and Scaling)...")
    X = normalize_landmarks(X)
    
    # 3. Shuffle before splitting (improves robustness since classes load sequentially)
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    # 4. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # 4. Build Model
    model = build_transformer_model(
        input_shape=(num_points, 3), 
        num_classes=26,
        num_transformer_blocks=4,
        head_size=256,
        num_heads=8,
        ff_dim=512,
        mlp_units=[256, 128],
        dropout=0.2,
        mlp_dropout=0.2
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    
    model.summary()
    
    # 5. Training Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"transformer_isl_{timestamp}.h5"
    checkpoint_path = f"models/{model_name}"
    
    callbacks = [
        PerformanceLogger(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy', 
            patience=20, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 6. Train
    print(f"Starting training model: {model_name}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.title('Landmark Transformer - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Landmark Transformer - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = f"models/history_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    print(f"Best model saved to {checkpoint_path}")
    
    # Save a convenience copy
    latest_link = "models/transformer_isl_latest.h5"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    import shutil
    shutil.copy2(checkpoint_path, latest_link)
    print(f"Latest model reference updated at {latest_link}")

    # Save metadata so inference knows the format
    import json
    meta = {
        "num_points": num_points,
        "num_features": num_features,
        "num_classes": 26,
        "timestamp": timestamp,
        "model_file": model_name
    }
    meta_path = f"models/training_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    # Also save as latest
    with open("models/training_meta_latest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved to {meta_path}")

if __name__ == "__main__":
    train_transformer()
