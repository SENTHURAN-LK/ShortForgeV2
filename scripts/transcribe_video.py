import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "input" / "test.wav"
OUTPUT = ROOT / "input" / "transcript.json"

# Make CUDA and cuDNN DLLs available to CTranslate2.
dll_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.28\bin\12.9\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
]

for dll_path in dll_paths:
    if Path(dll_path).is_dir():
        os.add_dll_directory(dll_path)
        os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel

def main():
    if not AUDIO.exists():
        raise FileNotFoundError(AUDIO)

    print("SHORTFORGE V2 - GPU TRANSCRIPTION", flush=True)
    print("Loading Faster-Whisper on NVIDIA GPU...", flush=True)

    started = time.perf_counter()

    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"
    )

    print("Model loaded. Transcribing audio...", flush=True)

    segments, info = model.transcribe(
        str(AUDIO),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )

    transcript = []
    word_count = 0

    for segment in segments:
        words = []

        for word in segment.words or []:
            words.append({
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "word": word.word
            })

        word_count += len(words)

        transcript.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "words": words
        })

        if len(transcript) % 20 == 0:
            print(
                f"Processed {len(transcript)} segments...",
                flush=True
            )

    OUTPUT.write_text(
        json.dumps(
            transcript,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    elapsed = time.perf_counter() - started

    print()
    print("GPU TRANSCRIPTION COMPLETE")
    print("Language:", info.language)
    print("Segments:", len(transcript))
    print("Words:", word_count)
    print("Processing time:", round(elapsed, 2), "seconds")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
