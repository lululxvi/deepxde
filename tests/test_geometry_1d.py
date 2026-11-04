"""Regression tests for interval boundary constraint factors."""

import unittest
from typing import Any

import numpy as np

from deepxde import backend as bkd
from deepxde.geometry import Interval


def _to_numpy(value: Any) -> np.ndarray:
    if bkd.backend_name == "tensorflow.compat.v1":
        with bkd.lib.Session() as session:
            return session.run(value)
    return bkd.to_numpy(value)


class TestIntervalBoundaryConstraintFactor(unittest.TestCase):
    def setUp(self) -> None:
        self.interval = Interval(-2.0, 3.0)
        self.points = bkd.as_tensor(
            np.array([[-2], [-0.75], [0.5], [1.75], [3]], dtype=np.float32)
        )

    def test_right_boundary(self) -> None:
        expected_factors = {
            "C0": [2, 1.5, 1, 0.5, 0],
            "C0+": [2, 1.5, 1, 0.5, 0],
            "Cinf": [4, 2.25, 1, 0.25, 0],
        }
        for smoothness, expected in expected_factors.items():
            with self.subTest(smoothness=smoothness):
                factor = self.interval.boundary_constraint_factor(
                    self.points, smoothness=smoothness, where="right"
                )
                np.testing.assert_allclose(
                    _to_numpy(factor), np.array(expected)[:, None]
                )

    def test_left_boundary(self) -> None:
        expected_factors = {
            "C0": [0, 0.5, 1, 1.5, 2],
            "C0+": [0, 0.5, 1, 1.5, 2],
            "Cinf": [0, 0.25, 1, 2.25, 4],
        }
        for smoothness, expected in expected_factors.items():
            with self.subTest(smoothness=smoothness):
                factor = self.interval.boundary_constraint_factor(
                    self.points, smoothness=smoothness, where="left"
                )
                np.testing.assert_allclose(
                    _to_numpy(factor), np.array(expected)[:, None]
                )

    def test_both_boundaries(self) -> None:
        expected_factors = {
            "C0": [0, 0.5, 1, 0.5, 0],
            "C0+": [0, 0.75, 1, 0.75, 0],
            "Cinf": [0, 0.5625, 1, 0.5625, 0],
        }
        for smoothness, expected in expected_factors.items():
            with self.subTest(smoothness=smoothness):
                factor = self.interval.boundary_constraint_factor(
                    self.points, smoothness=smoothness, where=None
                )
                np.testing.assert_allclose(
                    _to_numpy(factor), np.array(expected)[:, None]
                )

    def test_invalid_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "where must"):
            self.interval.boundary_constraint_factor(self.points, where="top")


if __name__ == "__main__":
    unittest.main()
