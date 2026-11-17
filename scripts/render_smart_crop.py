import json
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "input" / "test.mp4"
TRACKING = ROOT / "tracking" / "crop_path.json"
CAPTIONS = ROOT / "input" / "captions.ass"
HIGHLIGHTS = ROOT / "tracking" / "highlights.json"
OUTPUT = ROOT / "output" / "highlight_preview.mp4"

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

    highlight = json.loads(
        HIGHLIGHTS.read_text(encoding="utf-8")
    )["best_highlight"]

    start_time = float(highlight["start"])
    end_time = float(highlight["end"])

    if not 0 <= start_time < end_time:
        raise ValueError("Invalid highlight timestamps")

    start_frame = max(0, round(start_time * fps))
    end_frame = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        round(end_time * fps)
    )

    if end_frame <= start_frame:
        raise ValueError("Highlight contains no frames")

    actual_start = start_frame / fps

    # Shift ASS subtitle events to the trimmed timeline.
    def ass_time(value):
        cs = max(0, round(value * 100))
        h, rem = divmod(cs, 360000)
        m, rem = divmod(rem, 6000)
        s, c = divmod(rem, 100)
        return f"{h}:{m:02}:{s:02}.{c:02}"

    def parse_ass_time(value):
        h, m, rest = value.split(":")
        s, cs = rest.split(".")
        return (
            int(h) * 3600
            + int(m) * 60
            + int(s)
            + int(cs) / 100
        )

    subtitle_lines = []

    for line in CAPTIONS.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(",", 9)

            event_start = parse_ass_time(fields[1])
            event_end = parse_ass_time(fields[2])

            if event_end <= actual_start or event_start >= end_time:
                continue

            fields[1] = ass_time(
                max(0, event_start - actual_start)
            )
            fields[2] = ass_time(
                min(end_time, event_end) - actual_start
            )

            line = ",".join(fields)

        subtitle_lines.append(line)

    trimmed_captions = ROOT / "input" / "highlight_captions.ass"
    trimmed_captions.write_text(
        "\n".join(subtitle_lines) + "\n",
        encoding="utf-8-sig"
    )

    times = np.array([s["time"] for s in samples])
    positions = np.array([s["crop_left"] for s in samples])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    filtergraph = (
        "scale=1080:1920:flags=lanczos,"
        "setsar=1,"
        "ass=input/highlight_captions.ass"
    )

    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{crop_width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-ss", str(actual_start),
        "-t", str((end_frame - start_frame) / fps),
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

    frame_index = start_frame
    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        while frame_index < end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            time = frame_index / fps
            left = int(round(np.interp(time, times, positions)))
            left = max(0, min(max_left, left))

            cropped = frame[:, left:left + crop_width]

            process.stdin.write(cropped.tobytes())
            frame_index += 1
            processed += 1

            if processed % 150 == 0:
                print("Processed frames:", processed, flush=True)

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
    print("Frames:", processed)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
