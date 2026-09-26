import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "deepxde" / "utils" / "external.py"
SPEC = importlib.util.spec_from_file_location("deepxde_utils_external", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(external)


def test_pointset_vector_values_use_default_per_query_point():
    points = np.array([[0.0], [1.0]])
    values = np.array([[10.0, 11.0], [20.0, 21.0]])
    query = np.array([[0.0], [0.5], [1.0]])

    func = external.PointSet(points).values_to_func(
        values, default_value=np.array([-1.0, -2.0])
    )

    np.testing.assert_array_equal(
        func(query),
        np.array([[10.0, 11.0], [-1.0, -2.0], [20.0, 21.0]]),
    )
