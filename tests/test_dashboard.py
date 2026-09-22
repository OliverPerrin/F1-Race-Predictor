"""Smoke test the bundled deployment without local raw/processed datasets."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_bundled_history_and_upcoming_roster(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            deploy = Path(directory)
            shutil.copytree(root / 'src', deploy / 'src')
            shutil.copytree(root / 'data/predictions', deploy / 'data/predictions')
            metadata = json.loads((deploy / 'src/sample_data/metadata.json').read_text())
            previous = Path.cwd()
            try:
                os.chdir(deploy)
                app = AppTest.from_file(str(deploy / 'src/visualization.py'), default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('bundled dataset' in x.value for x in app.caption))
                app.multiselect[0].set_value([metadata['latest_year']]).run()
                app.selectbox[0].set_value(metadata['latest_race']).run()
                self.assertFalse(app.exception)
                latest_roster = set(app.dataframe[0].value.LastName)
                self.assertGreater(len(latest_roster), 0)
                app.radio[0].set_value('Upcoming weekend').run()
                self.assertFalse(app.exception)
                projected = app.dataframe[0].value
                self.assertEqual(set(projected.LastName), latest_roster)
                self.assertEqual(len(projected), len(latest_roster))
                self.assertTrue(projected.ActualPosition.isna().all())
                # Older seasons remain usable with the newly trained models.
                app.radio[0].set_value('Historical weekends').run()
                app.multiselect[0].set_value([2024]).run()
                self.assertFalse(app.exception)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
