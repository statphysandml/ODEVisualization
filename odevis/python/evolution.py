from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from odesolver.coordinates import Coordinates

from odevisualizationlib.evolution.stepper import RungaKutta4


class Evolution:
    def __init__(self, flow_equations: Any):
        from odevisualizationlib.modes import Evolution as VFEvolution
        self._evolution = VFEvolution(flow_equations._flow)

    def evolve(self, coordinates: Coordinates, stepper: Any=RungaKutta4(), start_t: float=0.0, end_t: Optional[float] = None, dt: float=0.01, n: Optional[int] = None, observer: Optional[Any] = None, equidistant_time_observations: bool=True, observe_every_ith_time_step: int=1):
        assert (n is None and end_t is not None) or (
                    n is not None and end_t is None), "Error: Note, either n or t1 needs to be find and only one of can be not None"
        if observer is None:
            if end_t is not None:
                self._evolution.evolve_const(stepper, coordinates.raw(), start_t, end_t, dt)
            else:
                self._evolution.evolve_n_steps(stepper, coordinates.raw(), start_t, dt, n)
        else:
            if end_t is not None:
                self._evolution.evolve_const(stepper, coordinates.raw(), start_t, end_t, dt, observer, equidistant_time_observations, observe_every_ith_time_step)
            else:
                self._evolution.evolve_n_steps(stepper, coordinates.raw(), start_t, dt, n, observer, equidistant_time_observations, observe_every_ith_time_step)
