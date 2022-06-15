from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable


from odesolver.coordinates import Coordinates


class KMeansClustering:
    def __init__(self, maximum_expected_number_of_clusters: int, upper_bound_for_min_distance: int, maximum_number_of_iterations: int = 1000):
        from odevisualizationlib.modes import KMeansClustering as VFKMeansClustering
        self._kmeans_clustering = VFKMeansClustering(maximum_expected_number_of_clusters, upper_bound_for_min_distance, maximum_number_of_iterations)

    def eval(self, coordinates, k: Optional[int] = None):
        if k is None:
            return self._kmeans_clustering.eval(coordinates.raw(), -1)
        else:
            return self._kmeans_clustering.eval(coordinates.raw(), k)
