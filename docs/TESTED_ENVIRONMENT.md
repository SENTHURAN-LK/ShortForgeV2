# ShortForge V2 - Tested Environment

## Platform
- Windows
- Python 3.11.9
- NVIDIA GeForce RTX 4070 Laptop GPU
- NVIDIA driver 621.24
- GPU memory: 8188 MiB

## GPU Libraries
- PyTorch 2.15.0+cu134
- CUDA runtime reported by PyTorch: 13.4
- CTranslate2 4.9.0
- Faster-Whisper 1.2.1
- Ultralytics 8.4.215

## Video Processing
- FFmpeg 9.0.2
- yt-dlp 2026.10.26

## Dependency Snapshot
See requirements-tested.txt for the complete tested
Python package versions.

## Important Notes
- NVIDIA GPU acceleration was verified on the test machine.
- CUDA-enabled PyTorch may require a dedicated package index.
- FFmpeg and NVIDIA drivers must be installed separately.
- The dependency snapshot does not guarantee compatibility
  with other operating systems or GPU configurations.
- GitHub Actions currently checks Python syntax only.
