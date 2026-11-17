import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "input" / "transcript.json"
OUTPUT = ROOT / "tracking" / "highlights.json"

MIN_DURATION = 15
MAX_DURATION = 30
MAX_CLIPS = 10
PADDING = 0.35

KEYWORDS = [
    "why", "how", "what", "because",
    "actually", "imagine", "never",
    "important", "interesting", "secret",
    "surprising", "but", "however"
]

def score_candidate(segments, start, end):
    text = " ".join(s["text"].strip() for s in segments)
    duration = end - start
    words = len(text.split())

    speech_duration = sum(
        max(0, s["end"] - s["start"])
        for s in segments
    )

    density = min(1, speech_duration / duration)

    score = min(words * 0.12, 12)
    score += text.count("?") * 4
    score += density * 12

    for keyword in KEYWORDS:
        if re.search(
            r"\b" + re.escape(keyword) + r"\b",
            text.lower()
        ):
            score += 2

    if 18 <= duration <= 27:
        score += 5

    return round(score, 2)

def main():
    segments = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    candidates = []

    for i in range(len(segments)):
        for j in range(i, len(segments)):
            start = max(
                0,
                segments[i]["start"] - PADDING
            )
            end = segments[j]["end"] + PADDING
            duration = end - start

            if duration > MAX_DURATION:
                break

            if duration < MIN_DURATION:
                continue

            selected = segments[i:j+1]

            candidates.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "score": score_candidate(
                    selected, start, end
                ),
                "text": " ".join(
                    s["text"].strip()
                    for s in selected
                )
            })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    chosen = []

    for candidate in candidates:
        overlaps = any(
            candidate["start"] < clip["end"]
            and candidate["end"] > clip["start"]
            for clip in chosen
        )

        if overlaps:
            continue

        chosen.append(candidate)

        if len(chosen) >= MAX_CLIPS:
            break

    chosen.sort(key=lambda x: x["start"])

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps({
            "best_highlight": (
                max(chosen, key=lambda x: x["score"])
                if chosen else None
            ),
            "highlights": chosen,
            "total_candidates": len(candidates)
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("MULTI-HIGHLIGHT SELECTION COMPLETE")
    print("Candidates:", len(candidates))
    print("Selected clips:", len(chosen))
    print()

    for i, clip in enumerate(chosen, 1):
        print(
            f"Clip {i:02}: "
            f"{clip['start']:.2f}s - "
            f"{clip['end']:.2f}s | "
            f"{clip['duration']:.2f}s | "
            f"Score: {clip['score']}"
        )

    print()
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
