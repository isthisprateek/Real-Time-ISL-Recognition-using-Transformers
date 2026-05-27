# Indian Sign Language (ISL) Landmark Transformer Architecture

This document describes the modern, landmark-based approach using a Transformer model. This method is significantly more robust than raw pixel-based CNNs as it focuses purely on the geometry of the hand.

---

## 📂 Folder Structure

```text
isl_project/
├── dataset/                    # All data storage
│   ├── raw/                    # Original images
│   │   └── merged/             # Optimized A-Z classes from Gesture Speech
│   └── landmarks.npz           # Extracted 21x3 landmarks (The training source)
├── models/                     # Saved trained models (.h5) and training plots
├── src/                        # Source code modules
│   ├── preprocessing/          # Data preparation logic
│   │   └── extract_landmarks.py # Uses MediaPipe Tasks to extract features
│   ├── training/               # Model architecture and training scripts
│   │   └── transformer_model.py # The Transformer Encoder definition
│   └── inference/              # Real-time prediction logic
├── train_transformer.py        # Main script to train the Transformer
├── hand_landmarker.task        # MediaPipe pre-trained model file
└── requirements_unpinned.txt   # Hardware/OS optimized dependencies
```

---

## 📄 Core Components

### 1. Feature Extraction (`extract_landmarks.py`)
*   **Input**: Normalized images (128x128).
*   **Logic**: Uses the **MediaPipe Tasks API** (`HandLandmarker`) to detect a hand and extract **21 landmarks**.
*   **Output**: A vector of 63 values (21 landmarks x 3 coordinates: X, Y, Z).
*   **Benefit**: Complete immunity to background light and skin tone.

### 2. Model Architecture (`transformer_model.py`)
*   **Type**: Transformer Encoder.
*   **Input Shape**: `(21, 3)` (21 joint nodes, each with 3 coordinates).
*   **Mechanism**: **Self-Attention** learns the spatial relationships between joints (e.g., how the tip of the thumb relates to the tip of the index finger).
*   **Complexity**: Much lighter and faster than CNNs (~500k-1M parameters).

### 3. Training Pipeline (`train_transformer.py`)
*   **Normalization**: Implements **Wrist-Centering** (making the wrist [0,0,0]) and **Unit Scaling**. This makes the model invariant to where the hand is in the frame or how close it is to the camera.
*   **Optimizer**: Adam with Learning Rate scheduling.

---

## 🔄 Data Pipeline Flow

1.  **Extract**: `extract_landmarks.py` converts the thousands of images into a single `landmarks.npz` file containing geometric data.
2.  **Normalize**: Landmarks are centered at the wrist and scaled.
3.  **Train**: The Transformer learns to classify the 21-node "hand graph" into one of 26 letters.
4.  **Live**: The webcam app extracts landmarks in real-time and passes them to the Transformer.

---

## 🛠️ Performance Benefits over CNN

| Metric | Old CNN Approach | New Transformer Approach |
| :--- | :--- | :--- |
| **Input** | 49,152 pixels | 63 geometric points |
| **Training Speed** | Slow (Hours) | Extremely Fast (Minutes) |
| **Robustness** | Sensitive to Light/BG | Immune to Environment |
| **Accuracy** | Prone to overfit pixels | Learns pure pose geometry |
