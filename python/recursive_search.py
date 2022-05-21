from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable

from odesolver.vfcoor import VFCoor
from odesolver.cube_index_path import CubeIndexPath

from odevisualizationlib.recursive_search import RecursiveSearch as VFRecursiveSearch


class RecursiveSearch:
    def __init__(
            self,
            maximum_recursion_depth: int,
            n_branches_per_depth: List[List],
            variable_ranges: List[List],
            criterion: Any,
            flow_equations: Any,
            number_of_cubes_per_gpu_call: int = 20000,
            maximum_number_of_gpu_calls: int = 1000
    ):
        self._recursive_search = VFRecursiveSearch(
            maximum_recursion_depth,
            n_branches_per_depth,
            variable_ranges,
            criterion,
            flow_equations._flow,
            flow_equations._jacobians,
            number_of_cubes_per_gpu_call,
            maximum_number_of_gpu_calls
        )

    def eval(self, memory_mode: str = "dynamic"):
        self._recursive_search.eval(memory_mode)

    def solutions(self, return_type="vfcoor"):
        if return_type == "vfcoor":
            return VFCoor(self._recursive_search.solutions())
        elif return_type == "cube_indices":
            return CubeIndexPath(self._recursive_search.leaves())
    
    # def project_cube_index_path_on_vertices(self, vertex_mode: str = "center")
    #     if vertex_mode == "center":
    #         pass
    #     elif vertex_mode == "reference":
    #         pass
    #     elif vertex_mode == "vertices":
    #         pass
    #     return VFCoor(...)