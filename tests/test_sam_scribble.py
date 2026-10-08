"""验证线提示的采样约束，线提示本身不生成或填补掩膜。"""

import unittest
import numpy as np
from core.sam_scribble import validate_scribbles, sample_scribbles


class ScribbleTests(unittest.TestCase):
    def test_native_arc_sampling_and_endpoints(self):
        strokes = validate_scribbles([{"points": [[1, 2], [11, 2], [11, 6]]}], 20, 20)
        samples, extent = sample_scribbles(strokes, np.array([10, 20]), 127)
        self.assertEqual(len(samples), 7)
        np.testing.assert_allclose(samples[0], [10, 40])
        np.testing.assert_allclose(samples[-1], [110, 120])
        np.testing.assert_allclose(samples[3], [100, 40])
        np.testing.assert_allclose(extent[1], [110, 40])

    def test_budget_retains_each_line_and_caps_density(self):
        strokes = [{"points": [[0, i], [10000, i]]} for i in range(3)]
        samples, _ = sample_scribbles(strokes, np.ones(2), 10)
        self.assertEqual(len(samples), 10)
        for i in range(3):
            path = samples[samples[:, 1] == i]
            self.assertEqual(path[0, 0], 0)
            self.assertEqual(path[-1, 0], 10000)
        samples, _ = sample_scribbles(strokes, np.ones(2), 128)
        self.assertEqual(len(samples), 96)
        with self.assertRaisesRegex(ValueError, "128"):
            sample_scribbles(strokes, np.ones(2), 5)

    def test_repeated_vertices_and_dense_mouse_events(self):
        short = [{"points": [[0, 0], [10, 0]]}]
        dense = [{"points": [[x, 0] for x in np.repeat(np.linspace(0, 10, 100), 2)]}]
        np.testing.assert_allclose(sample_scribbles(short, np.ones(2), 128)[0],
                                   sample_scribbles(dense, np.ones(2), 128)[0])

    def test_invalid_paths_are_rejected(self):
        for value in (None, {}, [{"points": [[1, 1]]}], [{"points": [[1, 1], [1, 1]]}],
                      [{"points": [[1, 1], [float("nan"), 2]]}],
                      [{"points": [[1, 1], [101, 2]]}], [{"points": [[-1, 1], [2, 2]]}],
                      [{"points": "无效"}], [{}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_scribbles(value, 100, 100)

    def test_limits_and_image_edges(self):
        stroke = {"points": [[0, 0], [100, 100]]}
        self.assertEqual(validate_scribbles([stroke], 100, 100), [stroke])
        for strokes in ([stroke] * 64, [{"points": [[i % 2, 0] for i in range(20001)]}]):
            with self.assertRaises(ValueError):
                validate_scribbles(strokes, 100, 100)


if __name__ == "__main__":
    unittest.main()
