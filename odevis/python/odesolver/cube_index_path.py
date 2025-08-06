from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable


from odevisualizationlib.recursive_search import Leaf


class CubeIndexPath:
    def __init__(self, cube_index_path: Any):
        self._cube_index_path = cube_index_path

    @classmethod
    def generate(cls, indices: Iterable):
        return cls(Leaf(indices))

    @property
    def depth(self):
        return self._cube_index_path.depth
    
    @property
    def indices(self):
        return self._cube_index_path.indices
    
    def __str__(self):
        return str(self.indices)