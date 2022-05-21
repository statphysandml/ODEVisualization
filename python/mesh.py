from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from odesolver.vfcoor import VFCoor


class Mesh:
    def __init__(self, n_branches: Iterable, variable_ranges: Iterable[Iterable],
        fixed_variables: Optional[Iterable[Iterable]] = None, iterative: bool = None, number_of_cubes_per_gpu_call: int = 20000):
        
        from odevisualizationlib.modes import Mesh as VFMesh
        if fixed_variables is None:
            self._mesh = VFMesh(n_branches, variable_ranges)
        else:
            self._mesh = VFMesh(n_branches, variable_ranges, fixed_variables)

    @property
    def n_branches(self):
        return self._mesh.n_branches

    @property
    def variable_ranges(self):
        return self._mesh.variable_ranges
    
    @property
    def fixed_variabels(self):
        if len(self._mesh.fixed_variables) == 0:
            return None
        else:
            return self._mesh.fixed_variables

    """Computes the vertices of the given grid - Note that the vertices are scanned from lower dimensions to
    higher dimensions, i.e. the scan starts in the first direction with fixed higher dimensions, and repreats this
    subsequently for respective values in higher dimensions."""
    def vertices(self, fixed_variable_idx: Optional[int] = 0):
        return VFCoor(self._mesh.eval(fixed_variable_idx))

    """Emulates the behaviour of np.mgrid applied on given VFCoor or on an evaluated grid based on the provided fixed_variable_idx"""
    def mgrid(self, fixed_variable_idx: Optional[int] = 0, data: Optional[VFCoor] = None):
        n_branch_idxs = [idx for idx, n_branch in enumerate(self.n_branches) if n_branch != 1]
        if data is None:
            return VFCoor(self._mesh.eval(fixed_variable_idx)).data_in_dim(n_branch_idxs).reshape((len(n_branch_idxs), *np.array(self.n_branches)[n_branch_idxs][::-1]))[::-1]
        else:
            return data.data_in_dim(n_branch_idxs).reshape((len(n_branch_idxs), *np.array(self.n_branches)[n_branch_idxs][::-1]))[::-1]