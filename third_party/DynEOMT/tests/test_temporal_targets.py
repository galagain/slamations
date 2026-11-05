import unittest
from types import SimpleNamespace

import torch
from detectron2.structures import BitMasks, Instances

from videomt.videomt import videomt


class TemporalTargetTest(unittest.TestCase):
    def test_class_is_read_from_first_visible_frame(self):
        model = videomt.__new__(videomt)
        torch.nn.Module.__init__(model)
        model.num_frames = 3
        model.register_buffer("pixel_mean", torch.zeros(3, 1, 1), persistent=False)

        frames = []
        frame_values = (
            ([10, -1], [2, 25]),
            ([-1, 11], [25, 7]),
            ([-1, -1], [25, 25]),
        )
        for ids, classes in frame_values:
            instances = Instances((4, 4))
            instances.gt_ids = torch.tensor(ids)
            instances.gt_classes = torch.tensor(classes)
            instances.gt_masks = BitMasks(torch.zeros((2, 4, 4), dtype=torch.bool))
            frames.append(instances)

        targets = [{"instances": frames}]
        images = SimpleNamespace(tensor=torch.zeros((3, 3, 4, 4)))
        prepared = model.prepare_targets(targets, images)

        self.assertEqual(prepared[0]["labels"].tolist(), [2, 7])
        self.assertEqual(
            prepared[0]["ids"].tolist(),
            [[10, -1, -1], [-1, 11, -1]],
        )


if __name__ == "__main__":
    unittest.main()
