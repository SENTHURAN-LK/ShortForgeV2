import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "input" / "transcript.json"
OUTPUT = ROOT / "input" / "captions.ass"

def timestamp(seconds):
    seconds = max(0, float(seconds))
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"

def escape_ass(text):
    return (
        str(text).replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )

def main():
    segments = json.loads(SOURCE.read_text(encoding="utf-8"))

    header = """[Script Info]
Title: ShortForge V2 Animated Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,76,&H00FFFFFF,&H0000D8FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,65,65,360,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    count = 0

    for segment in segments:
        words = segment.get("words", [])
        if not words:
            continue

        # Display up to four words together.
        for index in range(0, len(words), 4):
            group = words[index:index + 4]

            for active_index, active in enumerate(group):
                start = active.get("start")
                end = active.get("end")

                if start is None or end is None or end <= start:
                    continue

                caption_words = []

                for word_index, word in enumerate(group):
                    text = escape_ass(word.get("word", "").strip())

                    if word_index == active_index:
                        text = r"{\c&H0000D8FF&\fscx110\fscy110}" + text + r"{\rDefault}"

                    caption_words.append(text)

                caption = " ".join(caption_words)

                lines.append(
                    f"Dialogue: 0,{timestamp(start)},{timestamp(end)},"
                    f"Default,,0,0,0,,{caption}\n"
                )
                count += 1

    OUTPUT.write_text("".join(lines), encoding="utf-8-sig")

    print("CAPTION GENERATION SUCCESS")
    print("Caption events:", count)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
