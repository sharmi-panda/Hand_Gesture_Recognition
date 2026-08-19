import argparse
import time
import cv2
import config


def collect_samples(gesture_name: str, target_samples: int = 300):
    save_dir = config.RAW_DATA_DIR / gesture_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    collected = len(list(save_dir.glob("*.jpg")))
    recording = False

    print(f"\n[INFO] Collecting for: '{gesture_name}'")
    print("[INFO] Controls:")
    print("  - Press SPACE to start/pause recording")
    print("  - Press 'Q' or 'ESC' to exit early\n")

    try:
        while True:
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

            if recording and collected < target_samples:
                file_path = save_dir / f"{gesture_name}_{collected:04d}.jpg"
                cv2.imwrite(str(file_path), roi)
                collected += 1
                time.sleep(0.02)

            box_color = (0, 255, 0) if recording else (0, 165, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (0, 0), (w, 45), (25, 25, 25), -1)

            status_str = "RECORDING" if recording else "PAUSED (Press SPACE)"
            cv2.putText(
                frame,
                f"Class: {gesture_name} | {collected}/{target_samples} | {status_str}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                recording = not recording
            elif key in (ord("q"), 27):  # Q or ESC
                break

            if collected >= target_samples:
                print(f"[SUCCESS] Reached {target_samples} images for '{gesture_name}'.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hand gesture training images.")
    parser.add_argument("--gesture", type=str, required=True, choices=config.CLASSES)
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()

    collect_samples(args.gesture, args.samples)
