from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from odesolver.coordinates import Coordinates


class Jacobians:
    def __init__(self, elements: Union[Coordinates, Iterable, Iterable[Iterable]], dim: Optional[int] = None):
        from odevisualizationlib.modes import Jacobians as VFJacobians
        if isinstance(elements, Coordinates):
            self._jacobians = VFJacobians.from_devdat(elements.raw())
        elif isinstance(elements[0], Iterable):
            self._jacobians = VFJacobians.from_vec_vec(elements)
        else:
            self._jacobians = VFJacobians(elements, dim)

    def eval(self):
        self._jacobians.eval()

    @classmethod
    def from_vfcoor(cls, elements: Coordinates):
        return cls(elements=elements)

    @classmethod
    def from_flattened_data(cls, elements: Iterable, dim: int):
        return cls(elements=elements, dim=dim)

    @classmethod
    def from_data(cls, elements: Iterable[Iterable]):
        return cls(elements)

    @property
    def elements(self):
        return self._jacobians.elements()

    def jacobian(self, idx):
        return self._jacobians.jacobian(idx)

    def eigenvectors(self, idx):
        return self._jacobians.eigenvectors(idx)

    def eigenvalues(self, idx):
        return self._jacobians.eigenvalues(idx)
