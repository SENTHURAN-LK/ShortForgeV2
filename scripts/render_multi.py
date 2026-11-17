import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "input" / "test.mp4"
CAPTIONS = ROOT / "input" / "captions.ass"
HIGHLIGHTS = ROOT / "tracking" / "highlights.json"
TRACKING_DIR = ROOT / "tracking" / "clips"
OUTPUT_DIR = ROOT / "output" / "shorts"
CAPTION_DIR = ROOT / "input" / "clip_captions"


def parse_ass_time(value):
    h, m, rest = value.split(":")
    s, cs = rest.split(".")
    return (
        int(h) * 3600
        + int(m) * 60
        + int(s)
        + int(cs) / 100
    )


def ass_time(seconds):
    centiseconds = max(0, round(seconds * 100))
    h, remainder = divmod(centiseconds, 360000)
    m, remainder = divmod(remainder, 6000)
    s, cs = divmod(remainder, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def make_captions(number, start, end):
    lines = CAPTIONS.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    output_lines = []

    for line in lines:
        if line.startswith("Dialogue:"):
            fields = line.split(",", 9)

            event_start = parse_ass_time(fields[1])
            event_end = parse_ass_time(fields[2])

            if event_end <= start or event_start >= end:
                continue

            fields[1] = ass_time(
                max(0, event_start - start)
            )
            fields[2] = ass_time(
                min(end, event_end) - start
            )

            if parse_ass_time(fields[2]) <= parse_ass_time(fields[1]):
                continue

            line = ",".join(fields)

        output_lines.append(line)

    path = CAPTION_DIR / f"short_{number:02}.ass"
    path.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8-sig"
    )

    return path


def render_clip(number, clip):
    tracking_file = (
        TRACKING_DIR / f"crop_path_{number:02}.json"
    )

    tracking = json.loads(
        tracking_file.read_text(encoding="utf-8")
    )

    samples = tracking["samples"]

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    crop_width = int(tracking["crop_width"])
    max_left = width - crop_width

    start_frame = int(tracking["start_frame"])
    end_frame = int(tracking["end_frame"])

    if not 0 <= start_frame < end_frame <= total_frames:
        cap.release()
        raise ValueError("Invalid tracking frame range")

    actual_start = start_frame / fps
    duration = (end_frame - start_frame) / fps
    actual_end = actual_start + duration

    caption_path = make_captions(
        number, actual_start, actual_end
    )

    times = np.array(
        [sample["time"] for sample in samples],
        dtype=np.float64
    )
    positions = np.array(
        [sample["crop_left"] for sample in samples],
        dtype=np.float64
    )

    output = OUTPUT_DIR / f"short_{number:02}.mp4"

    caption_filter_path = (
        caption_path.relative_to(ROOT)
        .as_posix()
    )

    filtergraph = (
        "scale=1080:1920:flags=lanczos,"
        "setsar=1,"
        f"ass={caption_filter_path}"
    )

    command = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{crop_width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-ss", str(actual_start),
        "-t", str(duration),
        "-i", str(VIDEO),
        "-filter_complex",
        f"[0:v]{filtergraph}[v]",
        "-map", "[v]",
        "-map", "1:a:0",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "20",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output)
    ]

    print(
        f"Rendering short {number:02} | "
        f"{duration:.2f}s",
        flush=True
    )

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        cwd=str(ROOT)
    )

    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        for frame_index in range(start_frame, end_frame):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Video ended early at frame {frame_index}"
                )

            timestamp = frame_index / fps

            left = int(round(
                np.interp(timestamp, times, positions)
            ))

            left = max(0, min(max_left, left))

            cropped = np.ascontiguousarray(
                frame[:, left:left + crop_width]
            )

            process.stdin.write(cropped.tobytes())
            processed += 1

    except Exception:
        if process.stdin:
            process.stdin.close()
        process.kill()
        process.wait()
        raise

    finally:
        cap.release()

    process.stdin.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"FFmpeg failed on short {number:02}: "
            f"exit code {return_code}"
        )

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"Short {number:02} was not created"
        )

    print(
        f"Completed short {number:02} | "
        f"{processed} frames | "
        f"{output.stat().st_size / 1048576:.1f} MB",
        flush=True
    )


def main():
    highlights = json.loads(
        HIGHLIGHTS.read_text(encoding="utf-8")
    )["highlights"]

    if not highlights:
        raise RuntimeError("No highlights found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)

    print("SHORTFORGE V2 - MULTI-CLIP RENDER")
    print("Clips:", len(highlights))
    print("Encoder: NVIDIA NVENC")
    print("Resolution: 1080x1920")
    print()

    for number, clip in enumerate(highlights, 1):
        render_clip(number, clip)

    print()
    print("MULTI-CLIP RENDER COMPLETE")
    print("Videos:", len(highlights))
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
