# ShortForge V2 — Installation Guide

This guide describes the **tested Windows/NVIDIA workflow** for the `main.py` pipeline. ShortForge V2 is a working prototype; installation on a clean computer has not yet been independently verified.

## 1. Prerequisites

- Windows with an NVIDIA GPU and current compatible NVIDIA driver
- Python **3.11**, 64-bit
- Git (for cloning the repository)
- FFmpeg with `ffmpeg`, `ffprobe`, **NVENC**, and **libass/subtitles** support on `PATH`
- Internet access for downloading source videos, Python packages, and models
- NVIDIA CUDA/cuDNN runtime libraries compatible with your Faster-Whisper/CTranslate2 installation (see [GPU_SETUP.md](GPU_SETUP.md))
- Optional: Deno, which may help yt-dlp with YouTube JavaScript challenges

DaVinci Resolve is **not required** to run the tested pipeline. You can import completed MP4 files into Resolve afterward.

Check prerequisites in PowerShell:

```powershell
py -3.11 --version
git --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | Select-String h264_nvenc
ffmpeg -hide_banner -filters | Select-String subtitles
nvidia-smi
```

Install missing system tools from their official sources: [Python](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), [FFmpeg](https://ffmpeg.org/download.html), [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx), and optionally [Deno](https://deno.com/).

## 2. Get the project

Once the GitHub repository has been published, replace the placeholder URL with its real address:

```powershell
git clone https://github.com/YOUR_USERNAME/ShortForgeV2.git
cd ShortForgeV2
```

If you already have the project at `C:\ShortForgeV2`, simply run:

```powershell
cd C:\ShortForgeV2
```

## 3. Create the Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
```

Install the repository's declared Python packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Important:** The current `requirements.txt` is not a reproducible, version-locked environment specification. In particular, installing it alone does **not** guarantee a CUDA-enabled PyTorch build. Follow the next section before running the GPU pipeline.

## 4. Configure NVIDIA acceleration

Follow [GPU_SETUP.md](GPU_SETUP.md) to:

1. Install a CUDA-enabled PyTorch build compatible with your system.
2. Make the CUDA/cuDNN libraries needed by Faster-Whisper available to the Python process.
3. Confirm that PyTorch, Faster-Whisper, and YOLO work with the GPU.
4. Confirm FFmpeg supports NVIDIA H.264 encoding.

Do not assume that the CUDA version shown by `nvidia-smi` is the same as the CUDA runtime version bundled with PyTorch.

## 5. Verify the command-line entry point

```powershell
.\.venv\Scripts\python.exe .\main.py --help
```

The **tested** entry point is `main.py`, not the older `src\shortforge.py` implementation.

## 6. Run ShortForge V2

Use a YouTube video you own or have permission to process:

```powershell
.\.venv\Scripts\python.exe .\main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The pipeline downloads, normalizes, transcribes, ranks highlights, generates captions and crop paths, and renders vertical shorts.

Finished files appear in:

```text
output/shorts/short_01.mp4
output/shorts/short_02.mp4
...
```

**Data-loss warning:** The current `main.py` clears earlier generated shorts and selected intermediate files at the beginning of a new run. **Copy any outputs you want to keep elsewhere before running it again.** This behavior is scheduled for release hardening.

## 7. Check output files

```powershell
Get-ChildItem .\output\shorts\*.mp4
ffprobe -v error -show_entries stream=codec_name,width,height -show_entries format=duration -of json .\output\shorts\short_01.mp4
```

The verified test produced 10 H.264/AAC, 1080 × 1920 MP4s with durations of roughly 21–27 seconds. Results vary by source material and hardware.

## Troubleshooting

**`py -3.11` not found:** Install Python 3.11 x64 and reopen PowerShell.

**`ffmpeg` or `ffprobe` not found:** Add the FFmpeg `bin` directory to `PATH` and reopen PowerShell.

**`h264_nvenc` missing:** Install an FFmpeg build with NVENC support and verify your NVIDIA driver.

**`torch.cuda.is_available()` is `False`:** Check that the installed PyTorch wheel supports CUDA and that the NVIDIA driver is compatible. See [GPU_SETUP.md](GPU_SETUP.md).

**Faster-Whisper cannot load CUDA/cuDNN DLLs:** Verify the required runtime versions and DLL search paths; see [GPU_SETUP.md](GPU_SETUP.md).

**YouTube download errors:** Update yt-dlp and review any JavaScript runtime warning; availability and site behavior can change.

**No highlights selected:** Speech quality, duration, and transcript content affect the heuristic selection stage.

## Responsible use

Only download and repurpose videos you own or are authorized to use. Review generated clips before publishing.
