from collections import Counter, deque
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pyautogui

import config
from model import build_resnet_gesture_classifier


def execute_gesture_action(gesture: str, last_action_time: float, cooldown: float = 1.0) -> float:
    """Executes OS-level keyboard actions mapped to detected gestures."""
    current_time = time.time()
    if current_time - last_action_time < cooldown:
        return last_action_time

    if gesture == "open_palm":
        pyautogui.press("playpause")
        print("[ACTION] ✋ Open Palm -> Play/Pause")
        return current_time

    elif gesture == "thumbs_up":
        pyautogui.press("volumeup")
        print("[ACTION] 👍 Thumbs Up -> Volume Up (+)")
        return current_time - (cooldown - 0.2)

    elif gesture == "thumbs_down":
        pyautogui.press("volumedown")
        print("[ACTION] 👎 Thumbs Down -> Volume Down (-)")
        return current_time - (cooldown - 0.2)

    elif gesture == "heart":
        pyautogui.press("l")
        pyautogui.hotkey("alt", "shift", "b")
        print("[ACTION] ❤️ Heart -> Liked & Added to Playlist")
        return current_time + 1.5

    elif gesture == "snap":
        pyautogui.hotkey("shift", ".")
        print("[ACTION] ⚡ Snap -> 2X Speed (Shift + >)")
        return current_time + 0.8

    return last_action_time


def detect_hand_centroid(roi_frame):
    """Calculates horizontal hand centroid and contour area inside the ROI."""
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 2000:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"]), area
    return None, 0


def render_spotlight(frame, center_x, center_y, radius=190):
    """Draws a bright spotlight while dimming surrounding regions."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    dimmed = (frame * 0.2).astype(np.uint8)
    alpha = (mask / 255.0)[:, :, np.newaxis]
    spotlight_frame = (frame * alpha + dimmed * (1.0 - alpha)).astype(np.uint8)
    return spotlight_frame


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
    position_buffer = deque(maxlen=10)

    prev_frame_time = time.time()
    last_action_time = 0.0
    last_swipe_time = 0.0
    last_clap_time = 0.0
    prev_area = 0

    spotlight_active = False
    feedback_text = ""
    feedback_timer = 0.0

    print("\n[INFO] Gesture Controller Running. Press 'Q' or 'ESC' to exit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            box_size = 280
            top = (h - box_size) // 2
            bottom = top + box_size
            left = (w - box_size) // 2
            right = left + box_size

            roi = frame[top:bottom, left:right]
            curr_time = time.time()

            # --- 1. Hand Tracking, Swipe & Clap Detection ---
            hand_x, current_area = detect_hand_centroid(roi)

            # Clap Detection
            if prev_area > 2000 and (current_area - prev_area) > (0.65 * prev_area):
                if curr_time - last_clap_time > 1.2:
                    spotlight_active = not spotlight_active
                    feedback_text = f"👏 CLAP: SPOTLIGHT {'ON' if spotlight_active else 'OFF'}"
                    feedback_timer = curr_time
                    last_clap_time = curr_time
                    print(f"[ACTION] {feedback_text}")
            prev_area = current_area

            # Swipe Detection
            if hand_x is not None:
                position_buffer.append((hand_x, curr_time))

                if len(position_buffer) >= 4 and (curr_time - last_swipe_time > 1.0):
                    start_x, start_t = position_buffer[0]
                    dx = hand_x - start_x
                    dt = curr_time - start_t

                    if dt < 0.45:
                        if dx < -70:
                            pyautogui.press("prevtrack")
                            print("[ACTION] <<< SWIPE LEFT -> Previous Track")
                            feedback_text = "SWIPE LEFT: PREVIOUS SONG"
                            feedback_timer = curr_time
                            last_swipe_time = curr_time
                            position_buffer.clear()

                        elif dx > 70:
                            pyautogui.press("nexttrack")
                            print("[ACTION] >>> SWIPE RIGHT -> Next Track")
                            feedback_text = "SWIPE RIGHT: NEXT SONG"
                            feedback_timer = curr_time
                            last_swipe_time = curr_time
                            position_buffer.clear()

            # --- 2. Static Gesture Classification ---
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_roi).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred_idx = outputs.max(1)
                raw_gesture = config.CLASSES[pred_idx.item()]
                prediction_buffer.append(raw_gesture)

            current_gesture = Counter(prediction_buffer).most_common(1)[0][0]

            # Execute static action (no_gesture does nothing)
            if current_gesture != "no_gesture" and (curr_time - last_swipe_time > 0.8):
                last_action_time = execute_gesture_action(current_gesture, last_action_time)

            # Apply Spotlight
            if spotlight_active:
                frame = render_spotlight(frame, (left + right) // 2, (top + bottom) // 2, radius=190)

            # FPS
            fps = 1.0 / (curr_time - prev_frame_time + 1e-6)
            prev_frame_time = curr_time

            # --- 3. HUD Display ---
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 235, 0), 2)
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

            if curr_time - feedback_timer < 1.2:
                cv2.putText(frame, feedback_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
            elif current_gesture == "heart":
                cv2.putText(frame, "GESTURE: HEART (LIKED)", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 105, 255), 2, cv2.LINE_AA)
            elif current_gesture == "snap":
                cv2.putText(frame, "GESTURE: SNAP (2X SPEED)", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2, cv2.LINE_AA)
            else:
                color = (0, 255, 0) if current_gesture != "no_gesture" else (160, 160, 160)
                cv2.putText(frame, f"GESTURE: {current_gesture.upper()}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Hand Gesture & Motion Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_pipeline()
