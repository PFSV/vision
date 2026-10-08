import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vad'))
from rules import RuleDetector


class TemporalRulesTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'vad/config.json').read_text())
        self.config.update(max_people=0, restricted_region=[0.2, 0.2, 0.8, 0.8])
        self.person = dict(label='person', confidence=0.9, box=[30, 40, 70, 60])
        self.expected = ['crowding', 'possible_fall', 'restricted_area']

    def test_slow_continuous_observations_and_event_deduplication(self):
        detector = RuleDetector(self.config)
        for now in [0, 0.5, 1, 1.5]:
            self.assertEqual(detector.update([self.person], now, 100, 100), ([], []))
        self.assertEqual(detector.update([self.person], 2, 100, 100), (self.expected, self.expected))
        self.assertEqual(detector.update([self.person], 2.5, 100, 100), (self.expected, []))

    def test_missing_observation_resets_after_gap(self):
        detector = RuleDetector(self.config)
        detector.update([self.person], 0, 100, 100)
        detector.update([], 0.1, 100, 100)
        self.assertEqual(detector.update([self.person], 2, 100, 100), ([], []))

    def test_short_gap_and_confidence_filter(self):
        detector = RuleDetector(self.config)
        detector.update([self.person], 0, 100, 100)
        detector.update([], 0.1, 100, 100)
        detector.update([self.person], 0.2, 100, 100)
        self.assertEqual(detector.update([self.person], 2, 100, 100), (self.expected, self.expected))
        weak = dict(self.person, confidence=0.1)
        self.assertEqual(RuleDetector(self.config).update([weak], 0, 100, 100), ([], []))

    def test_invalid_region(self):
        self.config['restricted_region'] = [0.8, 0.2, 0.2, 0.8]
        with self.assertRaises(ValueError):
            RuleDetector(self.config)


if __name__ == '__main__':
    unittest.main()
