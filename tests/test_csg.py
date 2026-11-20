import numpy as np
import pytest
import deepxde as dde


def test_csg_union_disjoint():
    """Test CSGUnion with completely disjoint geometries."""
    g1 = dde.geometry.Interval(0, 1)
    g2 = dde.geometry.Interval(2, 3)
    union = dde.geometry.CSGUnion(g1, g2)
    
    # Inside checks
    assert union.inside(np.array([[0.5]]))[0]
    assert union.inside(np.array([[2.5]]))[0]
    assert not union.inside(np.array([[1.5]]))[0]
    
    # Boundary checks
    assert union.on_boundary(np.array([[0.0]]))[0]
    assert union.on_boundary(np.array([[1.0]]))[0]
    assert union.on_boundary(np.array([[2.0]]))[0]
    assert not union.on_boundary(np.array([[1.5]]))[0]

    # Periodic mapping (mapping from 0 to 1 for g1, 2 to 3 for g2)
    pt1 = np.array([[0.0]])
    assert np.allclose(union.periodic_point(pt1, component=0), np.array([[1.0]]))

@pytest.mark.xfail(reason="Known limitation: overlapping/touching CSG boundaries are not fully supported.")
def test_csg_union_touching():
    """Test CSGUnion with geometries touching at the boundary (x=1)."""
    g1 = dde.geometry.Interval(0, 1)
    g2 = dde.geometry.Interval(1, 2)
    union = dde.geometry.CSGUnion(g1, g2)
    
    # x=1 is a shared boundary, but in the union it is actually an interior point
    assert not union.on_boundary(np.array([[1.0]]))[0]
    
    # The left boundary of the union is 0, right is 2.
    # If a point at x=0 is mapped periodically, it should go to x=2.
    pt = np.array([[0.0]])
    mapped = union.periodic_point(pt, component=0)
    assert np.allclose(mapped, np.array([[2.0]]))

@pytest.mark.xfail(reason="Known limitation: CSGUnion bbox inherits from geom1, failing to map to the new merged bounds.")
def test_csg_union_overlapping():
    """Test CSGUnion with overlapping geometries."""
    g1 = dde.geometry.Rectangle([0, 0], [2, 2])
    g2 = dde.geometry.Rectangle([1, 0], [3, 2])
    union = dde.geometry.CSGUnion(g1, g2)
    
    # Domain is [0, 0] to [3, 2]
    # x=1 is inside the overlap region, should not be boundary
    assert not union.on_boundary(np.array([[1.0, 1.0]]))[0]
    
    # Map x=0 to x=3
    pt = np.array([[0.0, 1.0]])
    mapped = union.periodic_point(pt, component=0)
    assert np.allclose(mapped, np.array([[3.0, 1.0]]))

@pytest.mark.xfail(reason="Known limitation: CSGIntersection periodic mapping uses parent bounds instead of intersection bounds.")
def test_csg_intersection_overlapping():
    """Test CSGIntersection with overlapping geometries."""
    g1 = dde.geometry.Rectangle([0, 0], [2, 2])
    g2 = dde.geometry.Rectangle([1, 0], [3, 2])
    inter = dde.geometry.CSGIntersection(g1, g2)
    
    # Domain is [1, 0] to [2, 2]
    assert inter.inside(np.array([[1.5, 1.0]]))[0]
    assert not inter.inside(np.array([[0.5, 1.0]]))[0]
    
    # Map x=1 to x=2 (the new boundaries)
    pt = np.array([[1.0, 1.0]])
    mapped = inter.periodic_point(pt, component=0)
    assert np.allclose(mapped, np.array([[2.0, 1.0]]))


@pytest.mark.xfail(reason="Known issue: CSGDifference bbox does not represent the difference domain.")
def test_csg_difference_periodic():
    """Test CSGDifference periodic mapping (Expected to fail)."""
    g1 = dde.geometry.Interval(0, 2)
    g2 = dde.geometry.Interval(1, 2)
    diff = dde.geometry.CSGDifference(g1, g2)
    
    # Domain is [0, 1]. Mapping x=0 should go to x=1.
    pt = np.array([[0.0]])
    mapped = diff.periodic_point(pt, component=0)
    # This will fail because CSGDifference inherits bbox from g1 ([0, 2]), 
    # mapping x=0 to x=2 instead of x=1.
    assert np.allclose(mapped, np.array([[1.0]]))
