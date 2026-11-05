import json
import tempfile
import unittest
from pathlib import Path

from videomt.data_video.dynamic_targets import DynamicTargetStore


class DynamicTargetStoreTest(unittest.TestCase):
    def test_schema_v2_parsing_and_area_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target_directory = root / "vipseg" / "sequence"
            target_directory.mkdir(parents=True)
            payload = {
                "schema_version": 2,
                "frames": [
                    {
                        "frame": "00001.jpg",
                        "masks": [
                            {
                                "source_id": "10",
                                "track_id": "10",
                                "category_id": 2,
                                "area": 1,
                                "label": "static",
                                "probability": 0.1,
                            },
                            {
                                "source_id": "11",
                                "track_id": "11",
                                "category_id": 2,
                                "area": 3,
                                "label": "dynamic",
                                "probability": 0.9,
                            },
                            {
                                "source_id": "12",
                                "track_id": "12",
                                "category_id": 2,
                                "area": 100,
                                "label": "uncertain",
                                "probability": 0.5,
                            },
                        ],
                    }
                ],
            }
            (target_directory / "targets.json").write_text(json.dumps(payload))

            store = DynamicTargetStore(root, "vipseg", required=True)
            targets = store.load("sequence")
            values, valid = store.instance_values(targets, "11", ["00001.png"])
            category_values, category_valid = store.category_values(
                targets, 2, ["00001.jpg"]
            )

            self.assertEqual(values, [1.0])
            self.assertEqual(valid, [True])
            self.assertEqual(category_valid, [True])
            self.assertAlmostEqual(category_values[0], 0.75)

    def test_optional_missing_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = DynamicTargetStore(temporary_directory, "ovis", required=False)
            self.assertIsNone(store.load("missing"))


if __name__ == "__main__":
    unittest.main()
