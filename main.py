import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

INPUT = ROOT / "input"
TRACKING = ROOT / "tracking"
OUTPUT = ROOT / "output" / "shorts"

SOURCE = INPUT / "source.mp4"
VIDEO = INPUT / "test.mp4"
AUDIO = INPUT / "test.wav"


def run(command, stage):
    print()
    print("=" * 60)
    print(stage)
    print("=" * 60, flush=True)

    started = time.perf_counter()

    subprocess.run(
        [str(item) for item in command],
        cwd=str(ROOT),
        check=True
    )

    elapsed = time.perf_counter() - started

    print(
        f"COMPLETED: {stage} ({elapsed:.1f}s)",
        flush=True
    )


def script(filename, stage, *arguments):
    run(
        [
            PYTHON,
            ROOT / "scripts" / filename,
            *arguments
        ],
        stage
    )


def prepare():
    INPUT.mkdir(parents=True, exist_ok=True)
    TRACKING.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Remove stale intermediate files so a failed stage
    # cannot accidentally reuse a previous video's data.
    stale_files = [
        SOURCE,
        VIDEO,
        AUDIO,
        INPUT / "transcript.json",
        INPUT / "captions.ass",
        TRACKING / "highlights.json",
        TRACKING / "crop_path.json"
    ]

    stale_files.extend(
        (TRACKING / "clips").glob("crop_path_*.json")
    )

    stale_files.extend(
        OUTPUT.glob("short_*.mp4")
    )

    for path in stale_files:
        if path.exists():
            path.unlink()


def normalize_video():
    run(
        [
            "ffmpeg", "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", SOURCE,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            VIDEO
        ],
        "STAGE 2 - NORMALIZE VIDEO"
    )


def extract_audio():
    run(
        [
            "ffmpeg", "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", VIDEO,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            AUDIO
        ],
        "STAGE 3 - EXTRACT AUDIO"
    )


def verify():
    highlights_path = TRACKING / "highlights.json"

    highlights = json.loads(
        highlights_path.read_text(encoding="utf-8")
    )["highlights"]

    videos = sorted(OUTPUT.glob("short_*.mp4"))

    if len(videos) != len(highlights):
        raise RuntimeError(
            f"Expected {len(highlights)} shorts, "
            f"found {len(videos)}"
        )

    for video in videos:
        if video.stat().st_size == 0:
            raise RuntimeError(
                f"Empty output video: {video}"
            )

    print()
    print("=" * 60)
    print("SHORTFORGE V2 - PIPELINE COMPLETE")
    print("=" * 60)
    print("Highlights:", len(highlights))
    print("Rendered videos:", len(videos))
    print("Output directory:", OUTPUT)

    for video in videos:
        print(" ", video.name)


def main():
    parser = argparse.ArgumentParser(
        description="ShortForge V2 YouTube-to-Shorts automation"
    )

    parser.add_argument(
        "url",
        help="YouTube video URL"
    )

    args = parser.parse_args()

    started = time.perf_counter()

    print("SHORTFORGE V2")
    print("YouTube-to-Shorts Automation")
    print("Source:", args.url, flush=True)

    prepare()

    script(
        "download_video.py",
        "STAGE 1 - DOWNLOAD VIDEO",
        args.url
    )

    if not SOURCE.exists():
        raise RuntimeError(
            "Downloader did not create input/source.mp4"
        )

    normalize_video()
    extract_audio()

    script(
        "transcribe_video.py",
        "STAGE 4 - GPU TRANSCRIPTION"
    )

    script(
        "select_highlights.py",
        "STAGE 5 - SELECT HIGHLIGHTS"
    )

    script(
        "generate_captions.py",
        "STAGE 6 - GENERATE CAPTIONS"
    )

    script(
        "smart_crop_multi.py",
        "STAGE 7 - YOLO SMART TRACKING"
    )

    script(
        "render_multi.py",
        "STAGE 8 - RENDER SHORTS"
    )

    verify()

    elapsed = time.perf_counter() - started

    print(
        f"Total processing time: {elapsed:.1f} seconds"
    )


if __name__ == "__main__":
    main()
