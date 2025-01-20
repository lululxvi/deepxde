# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

__all__ = [
    "DictPointGeometry",
    "AbstractGeometry",
    "CSGDifference",
    "CSGIntersection",
    "CSGUnion",
    "Cuboid",
    "Disk",
    "Ellipse",
    "Geometry",
    "GeometryXTime",
    "Hypercube",
    "Hypersphere",
    "Interval",
    "PointCloud",
    "Polygon",
    "Rectangle",
    "Sphere",
    "StarShaped",
    "TimeDomain",
    "Triangle",
]

from .base import Geometry, AbstractGeometry, CSGDifference, CSGIntersection, CSGUnion, DictPointGeometry
from .geometry_1d import Interval
from .geometry_2d import Disk, Ellipse, Polygon, Rectangle, StarShaped, Triangle
from .geometry_3d import Cuboid, Sphere
from .geometry_nd import Hypercube, Hypersphere
from .pointcloud import PointCloud
from .timedomain import TimeDomain, GeometryXTime
