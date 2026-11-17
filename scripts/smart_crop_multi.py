import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "input" / "test.mp4"
HIGHLIGHTS = ROOT / "tracking" / "highlights.json"
OUTPUT_DIR = ROOT / "tracking" / "clips"

SAMPLE_RATE = 5

def smooth_positions(samples, crop_width, max_left, dt):
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
    desired = (forward + backward) / 2

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

    return samples

def main():
    highlights = json.loads(
        HIGHLIGHTS.read_text(encoding="utf-8")
    )["highlights"]

    if not highlights:
        raise RuntimeError("No highlights selected")

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Invalid source frame rate")

    crop_width = min(width, round(height * 9 / 16))
    max_left = width - crop_width
    step = max(1, round(fps / SAMPLE_RATE))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")

    print("MULTI-CLIP SMART TRACKING")
    print("Video:", width, "x", height)
    print("FPS:", fps)
    print("Highlights:", len(highlights))
    print("GPU: CUDA device 0", flush=True)

    try:
        for clip_number, clip in enumerate(highlights, 1):
            start_frame = max(
                0, round(float(clip["start"]) * fps)
            )
            end_frame = min(
                total_frames,
                round(float(clip["end"]) * fps)
            )

            if end_frame <= start_frame:
                raise ValueError(
                    f"Invalid frame range for clip {clip_number}"
                )

            previous_target = width / 2
            samples = []

            print(
                f"Analyzing clip {clip_number:02d}/{len(highlights)}...",
                flush=True
            )

            frame_indices = list(
                range(start_frame, end_frame, step)
            )

            if frame_indices[-1] != end_frame - 1:
                frame_indices.append(end_frame - 1)

            for frame_index in frame_indices:
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
                        "area": float(
                            (x2 - x1) * (y2 - y1)
                        )
                    })

                if people:
                    chosen = min(
                        people,
                        key=lambda p: (
                            abs(p["center"] - previous_target)
                            - min(
                                p["area"] / (width * height),
                                1
                            ) * 150
                        )
                    )
                    target = chosen["center"]
                    previous_target = target
                else:
                    target = previous_target

                samples.append({
                    "time": round(frame_index / fps, 4),
                    "target_x": round(float(target), 2),
                    "people": len(people)
                })

            if not samples:
                raise RuntimeError(
                    f"No frames analyzed for clip {clip_number}"
                )

            samples = smooth_positions(
                samples,
                crop_width,
                max_left,
                step / fps
            )

            output = (
                OUTPUT_DIR /
                f"crop_path_{clip_number:02d}.json"
            )

            output.write_text(
                json.dumps({
                    "source_width": width,
                    "source_height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "crop_width": crop_width,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "samples": samples
                }, indent=2),
                encoding="utf-8"
            )

            print(
                f"Clip {clip_number:02d}: "
                f"{len(samples)} samples saved",
                flush=True
            )

    finally:
        cap.release()

    print()
    print("MULTI-CLIP TRACKING COMPLETE")
    print("Tracking files:", len(highlights))
    print("Saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
