import unittest

from scripts.select_highlights import score_candidate


class TestHighlightScoring(unittest.TestCase):

    def test_score_is_numeric(self):
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is an interesting discovery."
            }
        ]

        score = score_candidate(segments, 0.0, 5.0)

        self.assertIsInstance(score, (int, float))

    def test_question_increases_score(self):
        base = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a discovery."
            }
        ]

        question = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a discovery?"
            }
        ]

        self.assertGreater(
            score_candidate(question, 0.0, 5.0),
            score_candidate(base, 0.0, 5.0)
        )

    def test_keyword_increases_score(self):
        base = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a discovery."
            }
        ]

        keyword = [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is an interesting discovery."
            }
        ]

        self.assertGreater(
            score_candidate(keyword, 0.0, 5.0),
            score_candidate(base, 0.0, 5.0)
        )


if __name__ == "__main__":
    unittest.main()
