import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import select_highlights as selector


class TestHighlightSelection(unittest.TestCase):

    def run_selection(self, segments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transcript.json"
            output = root / "highlights.json"

            source.write_text(
                json.dumps(segments),
                encoding="utf-8"
            )

            with patch.object(selector, "SOURCE", source), \
                 patch.object(selector, "OUTPUT", output):
                selector.main()

            return json.loads(
                output.read_text(encoding="utf-8")
            )

    def test_selected_clips_respect_duration_limits(self):
        segments = [
            {
                "start": float(i * 5),
                "end": float(i * 5 + 4),
                "text": "This is an interesting discovery."
            }
            for i in range(20)
        ]

        result = self.run_selection(segments)

        self.assertGreater(
            len(result["highlights"]), 0
        )

        for clip in result["highlights"]:
            self.assertGreaterEqual(
                clip["duration"], selector.MIN_DURATION
            )
            self.assertLessEqual(
                clip["duration"], selector.MAX_DURATION
            )

    def test_selected_clips_do_not_overlap(self):
        segments = [
            {
                "start": float(i * 5),
                "end": float(i * 5 + 4),
                "text": "Why is this discovery important?"
            }
            for i in range(20)
        ]

        result = self.run_selection(segments)
        clips = result["highlights"]

        for first, second in zip(clips, clips[1:]):
            self.assertLessEqual(
                first["end"], second["start"]
            )

    def test_maximum_clip_count(self):
        segments = [
            {
                "start": float(i * 5),
                "end": float(i * 5 + 4),
                "text": "Imagine this surprising discovery."
            }
            for i in range(100)
        ]

        result = self.run_selection(segments)

        self.assertLessEqual(
            len(result["highlights"]), selector.MAX_CLIPS
        )

    def test_empty_transcript(self):
        result = self.run_selection([])

        self.assertEqual(result["highlights"], [])
        self.assertIsNone(result["best_highlight"])
        self.assertEqual(result["total_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
