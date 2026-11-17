import sys
from pathlib import Path
import yt_dlp

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input"
OUTPUT = INPUT / "source.mp4"

def download_video(url):
    INPUT.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
        "outtmpl": str(INPUT / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "overwrites": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    if not OUTPUT.exists():
        raise RuntimeError(
            "Download completed, but source.mp4 was not found. "
            "Check the downloaded format."
        )

    print()
    print("VIDEO DOWNLOAD COMPLETE")
    print("Title:", info.get("title", "Unknown"))
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/download_video.py YOUTUBE_URL")
        sys.exit(1)

    download_video(sys.argv[1])
