import sys
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from visualize_dyneomt import DrawRecord, compose_frame, fit_text  # noqa: E402


class VisualizationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((60, 80, 3), 127, dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=bool)
        mask[10:50, 20:60] = True
        self.records = [
            DrawRecord(
                mask=mask,
                name="a_very_long_scientific_category_name",
                object_id="q123",
                color=(0, 200, 255),
                dynamic_probability=0.9,
                dynamic_label="dynamic",
            )
        ]
        self.args = Namespace(
            cell_width=320,
            cell_height=180,
            max_labels=5,
            dynamic_threshold=0.5,
            mode="grid",
        )

    def test_grid_and_prediction_dimensions(self):
        grid = compose_frame(
            self.image,
            self.records,
            self.records,
            self.records,
            self.records,
            "vipseg",
            "sequence",
            "frame.jpg",
            0,
            1,
            self.args,
        )
        self.assertEqual(grid.shape, (450, 640, 3))

        self.args.mode = "prediction"
        prediction = compose_frame(
            self.image,
            self.records,
            self.records,
            [],
            [],
            "vipseg",
            "sequence",
            "frame.jpg",
            0,
            1,
            self.args,
        )
        self.assertEqual(prediction.shape, (270, 640, 3))

    def test_long_text_is_fitted_to_panel(self):
        text, scale, width, _ = fit_text(
            "category_name_that_cannot_fit_without_resizing",
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            1,
            120,
        )
        self.assertLessEqual(width, 120)
        self.assertLessEqual(scale, 0.43)
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
