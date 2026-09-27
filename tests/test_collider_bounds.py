""" The broadphase bounds of a collider must enclose the collider as the narrowphase sees it """

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import bolt
from bolt.types import GeomType


def _capsule(radius: float, half_height: float):
    return bolt.convert_user_collider(bolt.UserGeomData(
        name="capsule", body_name="ground", geom_type=GeomType.CAPSULE,
        transform=wp.transform_identity(), size=wp.vec3(radius, half_height, radius)))


# The narrowphase (collision_primitive.py) takes a capsule's axis to be its local z, so its tips are at
# (0, 0, +/-(half_height + radius)) in the geom frame
CAPSULES = [(0.02, 0.05), (0.05, 0.02), (0.03, 0.0)]


@pytest.mark.parametrize("radius, half_height", CAPSULES)
def test_capsule_rbound_encloses_capsule(radius, half_height):
    """ The sphere and plane filters use rbound as the distance from the center to the farthest point """
    rbound = _capsule(radius, half_height).rbound
    assert rbound >= np.float32(half_height + radius), (
        f"rbound {rbound:.6g} < half_height + radius {half_height + radius:.6g}: "
        "contacts near the capsule's tips can be discarded")


@pytest.mark.parametrize("radius, half_height", CAPSULES)
def test_capsule_box_encloses_capsule(radius, half_height):
    """ The AABB and OBB filters use the box size as half-extents about the box center """
    center, size = _capsule(radius, half_height).aabb
    tip = np.array([0.0, 0.0, half_height + radius])
    for point in (tip, -tip):
        assert np.all(np.abs(point - np.array(center)) <= np.array(size) + 1e-7), (
            f"capsule tip {point.tolist()} lies outside the box of half-extents {[round(float(s), 4) for s in size]}: "
            "contacts near the capsule's tips can be discarded")
