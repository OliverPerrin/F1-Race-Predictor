"""Regression coverage for session cutoffs and current-season collection."""
import unittest
from unittest.mock import patch

import pandas as pd

from src.data_collection import F1DataCollector


class CollectorTests(unittest.TestCase):
    def test_default_years_include_current_season(self):
        self.assertEqual(F1DataCollector().years, list(range(2024, pd.Timestamp.now().year + 1)))

    def test_future_sprint_weekend_qualifying_is_not_loaded(self):
        # Sprint qualifying may already have happened, but grand prix qualifying has not.
        event = pd.DataFrame([{
            'EventFormat': 'sprint_qualifying', 'EventName': 'Future Grand Prix',
            'RoundNumber': 1, 'EventDate': '2099-01-01',
            'Session3': 'Sprint', 'Session3DateUtc': '2020-01-01',
            'Session4': 'Qualifying', 'Session4DateUtc': '2099-01-01',
            'Session5': 'Race', 'Session5DateUtc': '2099-01-02',
        }])
        with patch('src.data_collection.fastf1.get_event_schedule', return_value=event), \
             patch('src.data_collection.fastf1.get_session') as get_session:
            collector = F1DataCollector([2099])
            self.assertTrue(collector.collect_qualifying_results().empty)
            self.assertTrue(collector.collect_race_results().empty)
            get_session.assert_not_called()

    def test_empty_collection_cannot_replace_data(self):
        with self.assertRaises(ValueError):
            F1DataCollector([2026]).save_data(pd.DataFrame(), pd.DataFrame())


if __name__ == '__main__':
    unittest.main()
