"""Regression tests for polygon boundary arclength sampling."""

import unittest
from unittest.mock import patch

import numpy as np

from deepxde.geometry import Polygon


class TestPolygonBoundarySampling(unittest.TestCase):
    def setUp(self):
        self.polygon = Polygon([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])

    def sample_at(self, fractions):
        samples = np.array(fractions + [0.9375] * self.polygon.nvertices)[:, None]
        with patch("deepxde.geometry.geometry_2d.sample", return_value=samples):
            return self.polygon.random_boundary_points(len(fractions))

    def test_first_sample_can_fall_on_last_edge(self):
        points = self.sample_at([0.9375])
        np.testing.assert_allclose(points, [[0.5, 2]])
        self.assertTrue(self.polygon.on_boundary(points).all())

    def test_sparse_samples_can_skip_multiple_edges(self):
        cases = [
            ([0.9375, 0.0625], [[0, 1.5], [0.5, 2]]),
            ([0.4375, 0.9375], [[1.5, 0], [0.5, 2]]),
        ]
        for fractions, expected in cases:
            with self.subTest(fractions=fractions):
                points = self.sample_at(fractions)
                np.testing.assert_allclose(points, expected)
                self.assertTrue(self.polygon.on_boundary(points).all())

    def test_samples_on_consecutive_edges_are_unchanged(self):
        points = self.sample_at([0.125, 0.40625, 0.5625, 0.6875, 0.8125, 0.9375])
        np.testing.assert_allclose(
            points, [[0, 1], [1.25, 0], [2, 0.5], [1.5, 1], [1, 1.5], [0.5, 2]]
        )
        self.assertTrue(self.polygon.on_boundary(points).all())


if __name__ == "__main__":
    unittest.main()
