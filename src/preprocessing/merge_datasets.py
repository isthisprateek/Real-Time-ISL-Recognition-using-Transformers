"""
Landmark Extraction Script (Alternative entry point).
Uses the same LandmarkExtractor as extract_landmarks.py.
Both scripts produce IDENTICAL output format.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """
    Same extractor as extract_landmarks.py.
    Output: 126 values = Left(63) + Right(63), RAW coordinates, zero-padded.
    """
    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Organize by hand position (MUST match inference logic)
        hands = {}
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            wrist_x = hand_landmarks[0].x
            if wrist_x < 0.5:
                hand_type = "Left"
            else:
                hand_type = "Right"
            
            # Save RAW coordinates (normalization happens in training)
            landmark_list = []
            for lm in hand_landmarks:
                landmark_list.extend([lm.x, lm.y, lm.z])
            
            hands[hand_type] = landmark_list

        # Build fixed output: Left(63) + Right(63)
        final_features = []

        if "Left" in hands:
            final_features.extend(hands["Left"])
        else:
            final_features.extend([0.0] * 63)

        if "Right" in hands:
            final_features.extend(hands["Right"])
        else:
            final_features.extend([0.0] * 63)

        return np.array(final_features, dtype=np.float32)

    def process_dataset(self, raw_dir, output_file):
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            return

        VALID_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir() and d.name in VALID_CLASSES])
        data = []
        labels = []
        skipped = 0

        logger.info(f"Extracting landmarks from {raw_dir}")

        for cls_idx, cls_name in enumerate(classes):
            cls_dir = raw_dir / cls_name
            image_paths = list(cls_dir.glob("*"))

            logger.info(f"Processing class {cls_name} ({cls_idx+1}/{len(classes)})")

            for img_path in tqdm(image_paths, desc=f"Class {cls_name}"):
                landmarks = self.extract_landmarks(img_path)
                if landmarks is not None:
                    data.append(landmarks)
                    labels.append(cls_idx)
                else:
                    skipped += 1

        if not data:
            logger.error("No landmarks extracted. Check dataset and model path.")
            return

        X = np.array(data)
        y = np.array(labels)

        np.savez_compressed(output_file, X=X, y=y)

        logger.info(f"Saved to {output_file}")
        logger.info(f"Samples: {len(X)}, Skipped: {skipped}")
        logger.info(f"Features per sample: {X.shape[1]}")


if __name__ == "__main__":
    model_file = "/Users/prateek/Desktop/Sign Language/isl_project/hand_landmarker.task"

    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
    else:
        extractor = LandmarkExtractor(model_path=model_file)
        raw_path = "/Users/prateek/Desktop/Sign Language/isl_project/dataset/raw/merged"
        output_path = "/Users/prateek/Desktop/Sign Language/isl_project/dataset/landmarks.npz"
        extractor.process_dataset(raw_path, output_path)