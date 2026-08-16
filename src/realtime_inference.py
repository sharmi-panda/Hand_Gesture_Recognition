# src/realtime_inference.py
from collections import Counter, deque
import time
import cv2
import torch
from torchvision import transforms
from PIL import Image

import config
from model import build_resnet_gesture_classifier


def run_live_pipeline():
    if not config.MODEL_PATH.exists():
        print(f"[ERROR] Trained weights file not found at '{config.MODEL_PATH}'. Run train.py first!")
        return

    # Load Model
    model = build_resnet_gesture_classifier(freeze_backbone=False)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    prediction_buffer = deque(maxlen=7)
    prev_frame_time = time.time()

    print("\n[INFO] Starting Live Recognition. Press 'Q' or 'ESC' to exit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Center target box
            box_size = 250
            top = (h - box_size) // 2
            bottom = top + box_size
            left = (w - box_size) // 2
            right = left + box_size

            roi = frame[top:bottom, left:right]
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_roi).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred_idx = outputs.max(1)
                raw_gesture = config.CLASSES[pred_idx.item()]
                prediction_buffer.append(raw_gesture)

            current_gesture = Counter(prediction_buffer).most_common(1)[0][0]

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_frame_time + 1e-6)
            prev_frame_time = curr_time

            # HUD Display
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 235, 0), 2)
            cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"GESTURE: {current_gesture.upper()}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (w - 120, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_pipeline()