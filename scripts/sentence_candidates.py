from scripts.sentence_boundaries import extract_sentences


def generate_sentence_candidates(
    segments,
    min_duration=15,
    max_duration=30,
    padding=0.35
):
    """Generate highlight candidates using sentence boundaries."""

    if not segments:
        return []

    has_word_timestamps = all(
        segment.get("words")
        for segment in segments
    )

    if has_word_timestamps:
        units = extract_sentences(segments)

        # Do not end clips on an unfinished final sentence.
        if units and not units[-1]["text"].rstrip().endswith(
            (".", "!", "?")
        ):
            units = units[:-1]
    else:
        # Compatibility fallback for older transcripts.
        units = segments

    candidates = []

    for i in range(len(units)):
        for j in range(i, len(units)):
            start = max(0, units[i]["start"] - padding)
            end = units[j]["end"] + padding
            duration = end - start

            if duration > max_duration:
                break

            if duration < min_duration:
                continue

            selected = units[i:j + 1]

            candidates.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "text": " ".join(
                    unit["text"].strip()
                    for unit in selected
                ),
                "segments": selected
            })

    return candidates
