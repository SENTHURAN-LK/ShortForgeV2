# ShortForge V2

**Local YouTube-to-Shorts automation for Windows and NVIDIA GPUs**

ShortForge V2 converts long-form videos into vertical, captioned short clips using a local eight-stage processing pipeline. It combines Faster-Whisper transcription, heuristic highlight ranking, YOLO11n person detection, and FFmpeg rendering. No cloud AI API key is required.

> **Status:** Working prototype. The pipeline has completed an end-to-end test, but fresh-machine installation, automated tests, and release hardening are still in progress.

## Features

- Download a video from a YouTube URL using yt-dlp
- Transcribe speech with word timestamps using Faster-Whisper
- Rank potential highlights using explainable heuristic signals (not an LLM)
- Select up to 10 non-overlapping highlights, approximately 15–30 seconds each in the tested workflow
- Detect people using YOLO11n and generate smoothed portrait crop paths
- Burn animated word-by-word ASS subtitles into each clip
- Export 1080 × 1920 H.264/AAC MP4 videos with NVIDIA NVENC
- Run the tested workflow from a single Python command

## Processing pipeline

1. **Download** — fetch the source video with yt-dlp
2. **Normalize** — prepare the source with FFmpeg
3. **Extract audio** — produce audio for transcription
4. **Transcribe** — Faster-Whisper with word-level timestamps
5. **Select highlights** — score candidate speech windows and select non-overlapping clips
6. **Generate captions** — create animated ASS subtitle events
7. **Track subjects** — use YOLO11n to calculate smart-crop paths
8. **Render** — export captioned portrait clips using FFmpeg and NVIDIA NVENC

## Requirements

The tested setup uses:

- Windows with Python **3.11 (64-bit)**
- An NVIDIA CUDA-capable GPU (tested: **GeForce RTX 4070 Laptop GPU**)
- A working NVIDIA driver and compatible GPU-enabled PyTorch installation
- Compatible CUDA/cuBLAS and cuDNN runtime libraries for Faster-Whisper/CTranslate2
- FFmpeg and FFprobe available on `PATH`, with **NVENC** and **libass/subtitles** support
- Internet access for downloads and initial model retrieval

DaVinci Resolve is **not required** for the tested `main.py` workflow. Finished MP4s can be imported into Resolve manually.

## Quick start (Windows PowerShell)

Clone the repository once it is published (replace the placeholder with your actual repository URL):

```powershell
git clone https://github.com/YOUR_USERNAME/ShortForgeV2.git
cd ShortForgeV2
```

Create a Python 3.11 virtual environment and install the Python requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**GPU setup is a separate prerequisite.** Installing `requirements.txt` alone does not guarantee that CUDA-enabled PyTorch, CTranslate2, cuBLAS, or cuDNN will work. Follow the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) and [Faster-Whisper GPU guidance](https://github.com/SYSTRAN/faster-whisper#gpu) for compatible Windows versions. The project-specific setup guide is still being prepared.

Check the tools and GPU:

```powershell
py -3.11 --version
ffmpeg -version
ffprobe -version
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Run the tested pipeline:

```powershell
.\.venv\Scripts\python.exe .\main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Replace `VIDEO_ID` with the video you have permission to process.

## Output

Finished videos are written to:

```text
output/shorts/
  short_01.mp4
  short_02.mp4
  ...
  short_10.mp4
```

Intermediate downloads, transcripts, captions, highlight selections, and crop tracking data are stored in `input/` and `tracking/`. These directories and generated media are excluded from Git.

**Important — destructive cleanup:** The current `main.py` removes previously generated shorts and selected intermediate files when a new run starts. **Copy any outputs you want to keep somewhere else before running another video.** This behavior is scheduled for release hardening.

## Verified local test

An end-to-end run on a 12-minute, 43-second English-language podcast produced the following results on an RTX 4070 Laptop GPU:

| Metric | Observed result |
| --- | --- |
| End-to-end processing time | 238.4 seconds |
| Exported clips | 10 |
| Output resolution | 1080 × 1920 |
| Video codec | H.264 |
| Audio codec | AAC |
| Clip durations | Approximately 21–27 seconds |
| Transcription | Faster-Whisper GPU |
| Person tracking | YOLO11n GPU |
| Rendering | FFmpeg / NVIDIA NVENC |

These are **single-run observations**, not performance guarantees or a benchmark across multiple machines.

## Repository layout

```text
ShortForgeV2/
├── main.py                     # Tested pipeline entry point
├── scripts/
│   ├── download_video.py
│   ├── transcribe_video.py
│   ├── select_highlights.py
│   ├── generate_captions.py
│   ├── smart_crop_multi.py
│   └── render_multi.py
├── src/                        # Legacy implementation under review
├── docs/                       # Installation and GPU guides (planned)
├── tests/                      # Automated tests (planned)
├── .github/workflows/          # CI configuration (planned)
├── input/                      # Generated inputs (Git-ignored)
├── tracking/                   # Generated tracking data (Git-ignored)
├── output/                     # Rendered media (Git-ignored)
├── requirements.txt
├── run.ps1                     # Dependency installer; not the pipeline entry point
└── .gitignore
```

Older implementations are archived in `legacy/`, including `legacy/shortforge.py` and `legacy/scripts/`. These files are preserved for reference and are not part of the active pipeline. Use `main.py` to run ShortForge V2.

## Current limitations

- Highlight selection is heuristic, not full semantic understanding or a guarantee of engaging clips.
- Person detection does not establish who is speaking.
- Captions are burned into MP4s, not exported as editable Resolve title clips.
- The tested GPU workflow is Windows/NVIDIA-specific; CPU-only and cross-platform execution have not been verified for `main.py`.
- The existing installer and dependency list still need reproducibility testing on a clean machine.
- Previous output files are cleared at the beginning of each run.
- Always review generated clips and captions before publishing.

## Responsible use

Only download and repurpose videos you own or have permission to use. Follow applicable copyright rules and platform terms.

## Development status

ShortForge V2 is being prepared for an initial GitHub release. Documentation, dependency verification, tests, and safer output management are ongoing. A license has not yet been selected.
