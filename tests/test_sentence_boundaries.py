import unittest

from scripts.sentence_boundaries import extract_sentences


class TestSentenceBoundaries(unittest.TestCase):

    def test_multiple_sentences_in_one_segment(self):
        segments = [{
            "words": [
                {"start": 0.0, "end": 0.5, "word": " Hello"},
                {"start": 0.5, "end": 1.0, "word": " world."},
                {"start": 1.5, "end": 2.0, "word": " Why"},
                {"start": 2.0, "end": 2.5, "word": " now?"}
            ]
        }]

        result = extract_sentences(segments)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello world.")
        self.assertEqual(result[1]["text"], "Why now?")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[1]["end"], 2.5)

    def test_sentence_spanning_segments(self):
        segments = [
            {"words": [
                {"start": 0.0, "end": 0.4, "word": " This"},
                {"start": 0.4, "end": 0.8, "word": " is"}
            ]},
            {"words": [
                {"start": 1.0, "end": 1.5, "word": " important."}
            ]}
        ]

        result = extract_sentences(segments)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "This is important.")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 1.5)

    def test_unfinished_sentence_is_retained(self):
        segments = [{
            "words": [
                {"start": 0.0, "end": 0.5, "word": " An"},
                {"start": 0.5, "end": 1.0, "word": " unfinished"},
                {"start": 1.0, "end": 1.5, "word": " thought"}
            ]
        }]

        result = extract_sentences(segments)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["text"], "An unfinished thought"
        )

    def test_empty_transcript(self):
        self.assertEqual(extract_sentences([]), [])

    def test_exclamation_and_question_endings(self):
        segments = [{
            "words": [
                {"start": 0.0, "end": 0.5, "word": " Amazing!"},
                {"start": 0.7, "end": 1.0, "word": " Really?"}
            ]
        }]

        result = extract_sentences(segments)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
