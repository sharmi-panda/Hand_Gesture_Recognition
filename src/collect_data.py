# src/collect_data.py
import argparse
import sys
import time
from pathlib import Path
import cv2

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Dependency Hand Gesture Collector")
    parser.add_argument(
        "--gesture",
        "-g",
        type=str,
        default="thumbs_up",
        choices=config.CLASSES,
        help=f"Target gesture class to record. Options: {config.CLASSES}",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=300,
        help="Number of images to capture.",
    )
    return parser.parse_args()


def collect_samples():
    args = parse_args()
    save_folder = config.RAW_DATA_DIR / args.gesture
    save_folder.mkdir(parents=True, exist_ok=True)

    existing_images = list(save_folder.glob("*.jpg"))
    saved_count = len(existing_images)
    target_count = saved_count + args.samples

    print("=" * 55)
    print(f" DATA COLLECTOR (Pure OpenCV) -> Class: '{args.gesture}'")
    print(f" Existing Samples: {saved_count} | Target Total: {target_count}")
    print("=" * 55)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera feed unavailable.")
        sys.exit(1)

    is_recording = False

    try:
        while cap.isOpened() and saved_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # Define a fixed 250x250 target ROI box in the center of the frame
            box_size = 250
            top = (height - box_size) // 2
            bottom = top + box_size
            left = (width - box_size) // 2
            right = left + box_size

            # Crop ROI
            crop = frame[top:bottom, left:right]

            # Draw UI target box
            box_color = (0, 235, 0) if is_recording else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Header HUD
            status_text = f"REC ({saved_count}/{target_count})" if is_recording else "PAUSED (Press SPACE)"
            cv2.rectangle(frame, (0, 0), (width, 40), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"Class: {args.gesture.upper()} | {status_text}",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Place hand inside green box  |  SPACE: Rec  |  Q: Quit",
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Save frame when recording
            if is_recording and crop.size > 0:
                resized_crop = cv2.resize(crop, config.IMAGE_SIZE)
                filename = save_folder / f"{args.gesture}_{saved_count:04d}.jpg"
                cv2.imwrite(str(filename), resized_crop)
                saved_count += 1
                time.sleep(0.02)

            cv2.imshow("Hand Gesture Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                is_recording = not is_recording
            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[SUCCESS] Saved {saved_count} images in '{save_folder}'.\n")


if __name__ == "__main__":
    collect_samples()