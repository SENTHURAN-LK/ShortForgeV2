import re


SENTENCE_END = re.compile(r'[.!?]+["'']?$')


def extract_sentences(segments):
    """Convert Whisper word timestamps into sentence intervals."""
    sentences = []
    current_words = []

    def flush():
        if not current_words:
            return

        text = " ".join(
            word["word"].strip()
            for word in current_words
        )

        sentences.append({
            "start": round(current_words[0]["start"], 3),
            "end": round(current_words[-1]["end"], 3),
            "text": text
        })

        current_words.clear()

    for segment in segments:
        for word in segment.get("words", []):
            token = word["word"].strip()

            if not token:
                continue

            current_words.append(word)

            if SENTENCE_END.search(token):
                flush()

    flush()
    return sentences
