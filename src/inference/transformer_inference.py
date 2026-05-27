import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.transformer_model import build_transformer_model

# ============================================================
# IMPORTANT: This normalization must be IDENTICAL to the one 
# in train_transformer.py. Any difference = wrong predictions.
# ============================================================
def normalize_landmarks(landmarks):
    """
    Normalizes a single sample of landmarks.
    Each hand (21 points) is normalized independently:
    - Centered at its own wrist
    - Scaled to unit size
    - Zero-padded hands stay as zeros
    
    Input shape: (42, 3) for 2-hand data
    """
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

    return np.vstack([left, right])

def main():
    # 1. Configuration & Model Loading
    MODEL_PATH = os.path.join(project_root, 'models/transformer_isl_20260314_105705.h5')
    
    # Fallback: try to find the specific model if latest doesn't exist
    if not os.path.exists(MODEL_PATH):
        # Look for any transformer model
        models_dir = os.path.join(project_root, 'models')
        for f in sorted(os.listdir(models_dir), reverse=True):
            if f.startswith('transformer_isl_') and f.endswith('.h5'):
                MODEL_PATH = os.path.join(models_dir, f)
                break
    
    CONFIDENCE_THRESHOLD = 0.5
    STABILIZATION_WINDOW = 7
    
    # Detect model format from metadata
    NUM_POINTS = 42  # Default: 2-hand model
    meta_path = os.path.join(project_root, 'models/training_meta_latest.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        NUM_POINTS = meta.get("num_points", 42)
        print(f"Loaded metadata: {NUM_POINTS} landmarks per sample")
    
    print(f"Loading Transformer model weights from: {MODEL_PATH}")
    try:
        model = build_transformer_model(
            input_shape=(NUM_POINTS, 3), 
            num_classes=26,
            num_transformer_blocks=4,
            head_size=256,
            num_heads=8,
            ff_dim=512,
            mlp_units=[256, 128]
        )
        model.load_weights(MODEL_PATH)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize MediaPipe (using Solutions API for real-time video)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # 3. Prediction Stabilization
    prediction_buffer = deque(maxlen=STABILIZATION_WINDOW)
    
    # 4. Webcam Initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Real-time Transformer Inference Started ({NUM_POINTS} landmark points).")
    print("Press 'ESC' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip for selfie-view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        display_label = "Detecting..."
        display_confidence = 0.0
        combined_bbox = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # ============================================================
            # CRITICAL: Build the EXACT SAME feature vector as extraction.
            # Format: [Left_hand_21_landmarks | Right_hand_21_landmarks]
            # Each hand: 21 landmarks x 3 (x, y, z) = 63 values
            # Zero-pad if a hand is missing.
            # ============================================================
            
            # Step 1: Organize detected hands by handedness
            detected_hands = {}
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for hand_lms in results.multi_hand_landmarks:
                # Determine Left/Right by wrist x-position (not MediaPipe label)
                # This avoids mirror-flip inversion issues with cv2.flip + MediaPipe
                wrist_x = hand_lms.landmark[0].x
                if wrist_x < 0.5:
                    hand_label = "Left"
                else:
                    hand_label = "Right"
                
                # Draw landmarks on frame
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Extract RAW landmarks (same format as extraction script)
                landmarks = []
                for lm in hand_lms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Track bounding box
                    px, py = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, px)
                    y_min = min(y_min, py)
                    x_max = max(x_max, px)
                    y_max = max(y_max, py)
                
                detected_hands[hand_label] = landmarks
            
            # Step 2: Build fixed-size feature vector — MUST match extraction
            if NUM_POINTS == 42:
                # 2-hand model: [Left(63) + Right(63)]
                feature_vector = []
                if "Left" in detected_hands:
                    feature_vector.extend(detected_hands["Left"])
                else:
                    feature_vector.extend([0.0] * 63)
                
                if "Right" in detected_hands:
                    feature_vector.extend(detected_hands["Right"])
                else:
                    feature_vector.extend([0.0] * 63)
            else:
                # 1-hand model: take whichever hand is detected (prefer Right)
                if "Right" in detected_hands:
                    feature_vector = detected_hands["Right"]
                elif "Left" in detected_hands:
                    feature_vector = detected_hands["Left"]
                else:
                    feature_vector = [0.0] * 63

            # Step 3: Reshape to (NUM_POINTS, 3) and normalize
            landmarks_array = np.array(feature_vector, dtype=np.float32).reshape(NUM_POINTS, 3)
            normalized = normalize_landmarks(landmarks_array)
            
            # Step 4: Inference
            input_tensor = np.expand_dims(normalized, axis=0)
            prediction = model.predict(input_tensor, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Step 5: Bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            combined_bbox = (x_min, y_min, x_max, y_max)
            
            # Step 6: Stabilization
            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(class_id)
                if len(prediction_buffer) >= 3:
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    display_label = chr(most_common + 65)  # 0->A, 1->B, ...
                    # Use confidence of the stabilized class, not raw argmax
                    display_confidence = float(prediction[0][most_common])
            else:
                display_label = "Low Confidence"
                display_confidence = confidence

        # UI Overlay
        if combined_bbox:
            x_min, y_min, x_max, y_max = combined_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f"{display_label} ({display_confidence:.2f})"
            cv2.putText(frame, label_text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('ISL Transformer Inference', frame)

        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
