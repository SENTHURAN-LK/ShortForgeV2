import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "input" / "test.mp4"
OUTPUT = ROOT / "tracking" / "crop_path.json"
SAMPLE_RATE = 5

def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crop_width = min(width, round(height * 9 / 16))
    max_left = width - crop_width
    center_x = width / 2
    step = max(1, round(fps / SAMPLE_RATE))

    model = YOLO("yolo11n.pt")
    samples = []

    print("Analyzing video with YOLO...")
    print("Source:", width, "x", height)
    print("Crop width:", crop_width)
    print("Sample interval:", step, "frames")

    previous_target = center_x

    for frame_index in range(0, frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        result = model.predict(
            frame,
            device=0,
            classes=[0],
            conf=0.3,
            imgsz=640,
            verbose=False
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        people = []

        for x1, y1, x2, y2 in boxes:
            people.append({
                "center": float((x1 + x2) / 2),
                "width": float(x2 - x1),
                "area": float((x2 - x1) * (y2 - y1))
            })

        if people:
            # Prefer a person near the previously tracked position.
            # Area helps distinguish prominent subjects.
            chosen = min(
                people,
                key=lambda p: (
                    abs(p["center"] - previous_target)
                    - min(p["area"] / (width * height), 1) * 150
                )
            )
            target = chosen["center"]
            previous_target = target
        else:
            target = previous_target

        samples.append({
            "time": round(frame_index / fps, 3),
            "target_x": round(float(target), 2),
            "people": len(people)
        })

    cap.release()

    if not samples:
        raise RuntimeError("No video frames analyzed")

    # Cinematic smoothing: forward/backward filtering.
    targets = np.array(
        [s["target_x"] for s in samples],
        dtype=np.float64
    )

    alpha = 0.08

    def smooth(values):
        result = np.empty_like(values)
        result[0] = values[0]

        for i in range(1, len(values)):
            result[i] = (
                result[i - 1] * (1 - alpha)
                + values[i] * alpha
            )

        return result

    forward = smooth(targets)
    backward = smooth(forward[::-1])[::-1]

    # Combine both passes to reduce tracking delay.
    desired = (forward + backward) / 2

    # Limit camera velocity and acceleration.
    dt = step / fps
    max_speed = 120.0
    max_acceleration = 240.0

    position = desired[0]
    velocity = 0.0
    smoothed = [position]

    for target in desired[1:]:
        requested_velocity = np.clip(
            (target - position) / dt,
            -max_speed,
            max_speed
        )

        velocity_change = np.clip(
            requested_velocity - velocity,
            -max_acceleration * dt,
            max_acceleration * dt
        )

        velocity += velocity_change
        position += velocity * dt
        smoothed.append(position)

    for sample, target in zip(samples, smoothed):
        left = np.clip(
            target - crop_width / 2,
            0,
            max_left
        )
        sample["crop_left"] = round(float(left), 2)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({
            "source_width": width,
            "source_height": height,
            "fps": fps,
            "total_frames": frames,
            "crop_width": crop_width,
            "samples": samples
        }, indent=2),
        encoding="utf-8"
    )

    print("SMART CROP ANALYSIS COMPLETE")
    print("Samples:", len(samples))
    print("Crop range:", round(min(s["crop_left"] for s in samples)),
          "to", round(max(s["crop_left"] for s in samples)))
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
