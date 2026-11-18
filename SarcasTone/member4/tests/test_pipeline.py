"""Unit tests for Member 4 Multimodal Pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path

from member4.backend.pipeline import MultimodalPipeline


class TestMultimodalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = MultimodalPipeline()
        sheldon_wav = Path(__file__).resolve().parents[1] / "fixtures" / "preset_sheldon_sarcastic.wav"
        if sheldon_wav.is_file():
            cls.fixture_wav = sheldon_wav
        else:
            cls.fixture_wav = Path(__file__).resolve().parents[2] / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"

    def test_text_only_prediction(self) -> None:
        """Verify text-only inference with positive praise markers."""
        res = self.pipeline.predict(
            audio_path=None,
            text="Yeah, right, like that is ever going to happen.",
            context=["Do you think we will win the lottery?"],
            speaker="CHANDLER",
        )
        self.assertIn(res.label, ("sarcastic", "non_sarcastic"))
        self.assertIn("sarcastic", res.probabilities)
        self.assertIn("non_sarcastic", res.probabilities)
        self.assertEqual(res.modality_weights["text"], 1.0)
        self.assertEqual(res.modality_weights["audio"], 0.0)
        self.assertTrue(res.text_available)
        self.assertFalse(res.audio_available)
        self.assertGreater(len(res.explanation.top_text_tokens), 0)

    def test_multimodal_prediction_with_audio(self) -> None:
        """Verify full multimodal prediction with audio fixture."""
        if not self.fixture_wav.is_file():
            self.skipTest(f"Audio fixture {self.fixture_wav} not found.")

        res = self.pipeline.predict(
            audio_path=self.fixture_wav,
            text="It's just a privilege to watch your mind at work.",
            context=["I think I solved the puzzle."],
            speaker="SHELDON",
        )
        self.assertEqual(res.label, "sarcastic")
        self.assertGreater(res.confidence, 0.70)
        self.assertTrue(res.audio_available)
        self.assertTrue(res.text_available)
        self.assertGreater(res.modality_weights["audio"], 0.3)
        self.assertGreater(res.modality_weights["text"], 0.3)
        self.assertGreater(len(res.explanation.top_acoustic_features), 0)
        self.assertGreater(res.psid_dissonance_score, 0.40)

    def test_uncertainty_diagnostics(self) -> None:
        """Verify predictive entropy and uncertainty diagnostic flags."""
        res = self.pipeline.predict(
            audio_path=None,
            text="The weather is mild today.",
            context=None,
            speaker="LEONARD",
        )
        self.assertIsInstance(res.entropy, float)
        self.assertGreaterEqual(res.entropy, 0.0)
        self.assertIsInstance(res.uncertain, bool)


if __name__ == "__main__":
    unittest.main()
