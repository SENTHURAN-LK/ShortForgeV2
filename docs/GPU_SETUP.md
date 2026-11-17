# ShortForge V2 — NVIDIA GPU Setup

This document describes GPU requirements for the **tested Windows/NVIDIA `main.py` pipeline**. The project uses multiple GPU-dependent components, which must be verified **separately**.

## Tested configuration

The successful end-to-end run used:

- NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)
- Python 3.11 virtual environment
- CUDA-enabled PyTorch, reported as `2.15.0+cu134` in the tested environment
- CTranslate2 `4.9.0`
- Faster-Whisper `small` model with `device="cuda"` and `compute_type="float16"`
- Ultralytics YOLO11n with GPU inference
- FFmpeg H.264 NVENC encoding

These values describe **one working machine**, not a universally compatible set of install commands. PyTorch wheel availability and CUDA/cuDNN compatibility change over time.

## 1. Check the NVIDIA driver

```powershell
nvidia-smi
```

Confirm the GPU is listed. If not, install or update the driver from [NVIDIA](https://www.nvidia.com/Download/index.aspx) and restart.

**Note:** The `CUDA Version` displayed by `nvidia-smi` indicates driver-supported CUDA capability; it does not prove that a matching CUDA Toolkit or cuDNN runtime is installed.

## 2. Install CUDA-enabled PyTorch

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose **Windows**, **Pip**, **Python**, and an available **CUDA** build appropriate for your driver and supported Python version. Run the command provided by PyTorch inside the project's virtual environment. Do not blindly reuse a wheel index from another machine.

Check the result:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

Continue only after `CUDA available: True` if you intend to use the GPU pipeline.

## 3. Configure Faster-Whisper CUDA libraries

Faster-Whisper uses **CTranslate2**, which has its own NVIDIA runtime requirements. Consult the [Faster-Whisper GPU documentation](https://github.com/SYSTRAN/faster-whisper#gpu) and the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html) for the CUDA/cuDNN versions required by your installed CTranslate2 build.

For the tested setup, CUDA 12-family cuBLAS libraries and cuDNN 9 libraries were configured. The machine also had CUDA and cuDNN DLL directories made available to the Python process. The exact paths are **machine-specific**; do not copy someone else's installation paths without checking them.

Check your installed package version:

```powershell
.\.venv\Scripts\python.exe -c "import ctranslate2; print('CTranslate2:', ctranslate2.__version__); print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

Then test GPU model loading:

```powershell
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('Faster-Whisper CUDA model load: OK')"
```

Model loading is a **smoke test**, not a complete transcription test. A full pipeline run is needed to confirm actual transcription.

If a required DLL cannot be found, inspect the relevant CUDA/cuDNN installation and the process `PATH`. Restart PowerShell after changing environment variables. Avoid mixing unrelated CUDA/cuDNN versions in the same DLL search path.

## 4. Verify YOLO11n GPU inference

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; import torch; model=YOLO('yolo11n.pt'); print('YOLO11n loaded; CUDA:', torch.cuda.is_available())"
```

This confirms model loading, not GPU inference. To verify inference, run the following **only if you have a local test image**:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt').predict('test.jpg', device=0, verbose=False); print('YOLO GPU inference: OK')"
```

Replace `test.jpg` with an existing image path. The model weights are downloaded automatically if needed; `*.pt` files are excluded from Git.

## 5. Verify FFmpeg NVIDIA encoding and subtitles

```powershell
ffmpeg -hide_banner -encoders | Select-String h264_nvenc
ffmpeg -hide_banner -filters | Select-String subtitles
```

Both checks should return matching entries. The pipeline requires the subtitle filter for burning captions and NVENC support for the tested rendering path.

## 6. Run an end-to-end check

First **back up any existing `output/shorts/` videos**. The current pipeline clears previous generated shorts and selected intermediate files when starting a new run.

```powershell
.\.venv\Scripts\python.exe .\main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Use a video you own or have permission to process. Verify the exported files in `output/shorts/`.

## Troubleshooting notes

- **PyTorch CUDA unavailable:** A CPU-only PyTorch installation may have been installed by the dependency resolver; reinstall a compatible CUDA build using the official selector.
- **CTranslate2 sees zero CUDA devices:** Confirm driver/runtime compatibility and required libraries.
- **Faster-Whisper fails while PyTorch works:** These use different GPU backends; check CTranslate2's own CUDA/cuDNN requirements.
- **YOLO runs on CPU:** Verify `torch.cuda.is_available()` and that inference explicitly selects `device=0`.
- **NVENC encoder fails:** Check FFmpeg build capabilities, NVIDIA driver, and GPU support.
- **DLL loading blocked on Windows:** Security or application-control policies may block DLLs; investigate the Windows event logs and applicable administrator policies. Do not disable security controls as a generic installation step.

## Scope

This guide documents a known working configuration, not a promise of compatibility with every NVIDIA GPU, driver, or Windows version. Fresh-machine setup remains to be validated before a stable release.
