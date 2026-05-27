import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.image_utils import ImagePreprocessor

def main():
    # 1. Configuration & Model Loading
    MODEL_PATH = os.path.join(project_root, 'models/best_model_20260313_181340.h5')
    TARGET_SIZE = (128, 128)
    CONFIDENCE_THRESHOLD = 0.6
    STABILIZATION_WINDOW = 10
    
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize MediaPipe & Preprocessor
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    preprocessor = ImagePreprocessor(target_size=TARGET_SIZE)
    
    # 3. Prediction Stabilization
    prediction_buffer = deque(maxlen=STABILIZATION_WINDOW)
    
    # 4. Webcam Initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Real-time Inference Started. Press 'ESC' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert the BGR image to RGB before processing
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        display_label = "Detecting..."
        display_confidence = 0.0
        bbox = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Compute Bounding Box
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y
                
                # Add Padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                bbox = (x_min, y_min, x_max, y_max)
                
                # Crop Hand Region
                hand_img = frame[y_min:y_max, x_min:x_max]
                
                if hand_img.size != 0:
                    # Preprocess using ImagePreprocessor
                    # ImagePreprocessor expects BGR (as read by cv2.imread usually)
                    # and internal logic normally handles BGR to RGB if needed, 
                    # but ImagePreprocessor.resize_image and normalize_image work on arrays directly.
                    
                    # Ensure RGB for model (training was RGB)
                    if hand_img.shape[0] < 20 or hand_img.shape[1] < 20:
                        continue

                    hand_img = cv2.resize(hand_img, (128,128))
                    hand_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    
                    resized_img = preprocessor.resize_image(hand_rgb)
                    normalized_img = preprocessor.normalize_image(resized_img)
                    
                    # Add batch dimension: (1, 128, 128, 3)
                    input_tensor = np.expand_dims(normalized_img, axis=0)
                    
                    # Inference
                    prediction = model.predict(input_tensor, verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Stabilization & Filtering
                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_buffer.append(class_id)
                        
                        # Majority Vote
                        if len(prediction_buffer) == STABILIZATION_WINDOW:
                            most_common = Counter(prediction_buffer).most_common(1)[0][0]
                            display_label = chr(most_common + 65)
                            display_confidence = confidence
                    else:
                        display_label = "Detecting..."
                        display_confidence = confidence

        # UI Overlay
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            label_text = f"{display_label} ({display_confidence:.2f})"
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Output
        cv2.imshow('ISL Real-time Recognition', frame)

        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
