import unittest
from collections import defaultdict

from videomt.data_video.dynamic_evaluation import dynamic_metrics


class DynamicMetricsTest(unittest.TestCase):
    def test_perfect_conditional_classification_and_imperfect_coverage(self):
        stats = defaultdict(
            float,
            {
                "tp": 1,
                "tn": 1,
                "fp": 0,
                "fn": 0,
                "matched": 2,
                "gt": 3,
                "pred": 4,
                "joint_correct": 2,
                "iou_sum": 1.5,
            },
        )
        metrics = dynamic_metrics(stats, labels=[0, 1], scores=[0.1, 0.9])

        self.assertAlmostEqual(metrics["Dyn-MCC"], 100.0)
        self.assertAlmostEqual(metrics["Dyn-BAcc"], 100.0)
        self.assertAlmostEqual(metrics["Dyn-AUROC"], 100.0)
        self.assertAlmostEqual(metrics["MaskCoverage"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["Matched-IoU"], 75.0)
        self.assertAlmostEqual(metrics["DynSeg-Precision"], 50.0)
        self.assertAlmostEqual(metrics["DynSeg-Recall"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["DynSeg-F1"], 400.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
