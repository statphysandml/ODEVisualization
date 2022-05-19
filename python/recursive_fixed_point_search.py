from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable

from odesolver.vfcoor import VFCoor

from odevisualizationlib.modes import FixedPointSearch


class RecursiveFixedPointSearch:
    def __init__(
            self,
            maximum_recursion_depth: int,
            n_branches_per_depth: List[List],
            variable_ranges: List[List],
            flow_equations: Any,
            number_of_cubes_per_gpu_call: int = 20000,
            maximum_number_of_gpu_calls: int = 1000
    ):
        self._fixed_point_search = FixedPointSearch(
            maximum_recursion_depth,
            n_branches_per_depth,
            variable_ranges,
            flow_equations._flow,
            flow_equations._jacobians,
            ""
        )

    def eval(self, memory_mode: str = "dynamic"):
        self._fixed_point_search.eval(memory_mode)

    def fixed_points(self):
        return VFCoor(self._fixed_point_search.fixed_points())