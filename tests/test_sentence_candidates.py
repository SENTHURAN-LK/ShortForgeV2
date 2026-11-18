import unittest

from scripts.sentence_candidates import generate_sentence_candidates


class TestSentenceCandidates(unittest.TestCase):

    def test_empty_transcript(self):
        self.assertEqual(
            generate_sentence_candidates([]), []
        )

    def test_duration_limits(self):
        segments = [{
            "words": [
                {
                    "start": float(i * 5),
                    "end": float(i * 5 + 4),
                    "word": " Discovery."
                }
                for i in range(20)
            ]
        }]

        candidates = generate_sentence_candidates(segments)

        self.assertGreater(len(candidates), 0)

        for candidate in candidates:
            self.assertGreaterEqual(
                candidate["duration"], 15
            )
            self.assertLessEqual(
                candidate["duration"], 30
            )

    def test_sentence_boundary_alignment(self):
        segments = [{
            "words": [
                {"start": 0.0, "end": 5.0, "word": " First."},
                {"start": 6.0, "end": 11.0, "word": " Second."},
                {"start": 12.0, "end": 17.0, "word": " Third."},
                {"start": 18.0, "end": 23.0, "word": " Fourth."}
            ]
        }]

        candidates = generate_sentence_candidates(segments)

        self.assertTrue(candidates)

        for candidate in candidates:
            self.assertTrue(
                candidate["text"].endswith(".")
            )

    def test_legacy_segment_fallback(self):
        segments = [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "First segment."
            },
            {
                "start": 10.0,
                "end": 20.0,
                "text": "Second segment."
            }
        ]

        candidates = generate_sentence_candidates(segments)

        self.assertTrue(candidates)
        self.assertEqual(
            candidates[0]["text"],
            "First segment. Second segment."
        )

    def test_unfinished_final_sentence_excluded(self):
        segments = [{
            "words": [
                {"start": 0.0, "end": 8.0, "word": " First."},
                {"start": 8.5, "end": 17.0, "word": " Second."},
                {"start": 17.5, "end": 26.0, "word": " unfinished"}
            ]
        }]

        candidates = generate_sentence_candidates(segments)

        self.assertTrue(candidates)

        for candidate in candidates:
            self.assertNotIn(
                "unfinished", candidate["text"]
            )


if __name__ == "__main__":
    unittest.main()
