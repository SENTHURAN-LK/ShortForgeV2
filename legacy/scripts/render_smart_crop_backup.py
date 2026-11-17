import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "input" / "test.mp4"
TRACKING = ROOT / "tracking" / "crop_path.json"
CAPTIONS = ROOT / "input" / "captions.ass"
OUTPUT = ROOT / "output" / "smart_crop_preview.mp4"

def main():
    data = json.loads(TRACKING.read_text(encoding="utf-8"))
    samples = data["samples"]

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    crop_width = int(data["crop_width"])
    max_left = width - crop_width

    times = np.array([s["time"] for s in samples])
    positions = np.array([s["crop_left"] for s in samples])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    filtergraph = (
        "scale=1080:1920:flags=lanczos,"
        "setsar=1,"
        "ass=input/captions.ass"
    )

    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{crop_width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(VIDEO),
        "-filter_complex", f"[0:v]{filtergraph}[v]",
        "-map", "[v]",
        "-map", "1:a:0",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "20",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT)
    ]

    print("Starting smart-crop render...", flush=True)
    print("Output:", OUTPUT, flush=True)

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        cwd=str(ROOT)
    )

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time = frame_index / fps
            left = int(round(np.interp(time, times, positions)))
            left = max(0, min(max_left, left))

            cropped = frame[:, left:left + crop_width]

            process.stdin.write(cropped.tobytes())
            frame_index += 1

            if frame_index % 150 == 0:
                print("Processed frames:", frame_index, flush=True)

    except BrokenPipeError:
        raise RuntimeError("FFmpeg stopped accepting video frames")
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()

    result = process.wait()

    if result != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {result}")

    print("SMART CROP RENDER COMPLETE")
    print("Frames:", frame_index)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
