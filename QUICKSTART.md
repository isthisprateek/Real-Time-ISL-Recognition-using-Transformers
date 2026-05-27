# 🚀 Quick Start Guide - ISL Recognition System


1. **Model**: Transformer Encoder (Geometric Pattern Recognition)
2. **Input**: Hand Landmarks (X, Y, Z coordinates) via MediaPipe
3. **Accuracy**: ~94% on geometric shapes
4. **Performance**: Real-time (20-50 FPS) on CPU

---

## 🛠️ Step-by-Step Setup

### 1. Installation
```bash
# It is recommended to use Python 3.11
pip install -r requirements.txt
```

### 2. Extract Geometric Patterns
Converts your image dataset into lightweight landmark files:
```bash
python src/preprocessing/extract_landmarks.py
```

### 3. Train the Transformer
```bash
python train_transformer.py
```
*Wait ~10-15 minutes. This generates `models/transformer_isl_*.h5`.*

### 4. Run Real-Time Inference
```bash
python src/inference/transformer_inference.py
```

---

## 🎮 Controls (In-App)
- **ESC**: Quit the application.
- **Hand Visibility**: Ensure your hands are fully visible within the camera frame for the best landmarks.

---

## 🧠 Why This Approach?
Unlike the old CNN method which looked at pixels, this Transformer version looks at **hand anatomy**. 

| Feature | Transformer (Current) | CNN (Old) |
| :--- | :--- | :--- |
| **Data Size** | 63 points | 49,152 pixels |
| **Robustness** | Works in dark/cluttered rooms | Needs clean background |
| **Logic** | Understands joint relationships | Guesses based on pixel colors |

---

## 📂 Key Files
- `src/preprocessing/extract_landmarks.py`: The "Eyes" - extracts landmarks.
- `src/training/transformer_model.py`: The "Brain" - Transformer architecture.
- `src/inference/transformer_inference.py`: The "App" - Connects webcam to the brain.

---

## 🚀 Next Steps
1. **Try 1 Hand**: Test symbols like 'A', 'E', 'S'.
2. **Try 2 Hands**: Test complex symbols where hands interact.
3. **Environment**: Ensure your lighting is consistent for MediaPipe tracking.

Happy signing! 🤟
