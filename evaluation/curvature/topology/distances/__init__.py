from .base import TopologicalDistance
from .image import ImageDistance
from .landscape import LandscapeDistance

supported_distances = {"landscape": LandscapeDistance, "image": ImageDistance}

__all__ = ["TopologicalDistance", "ImageDistance", "LandscapeDistance"]
