from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from odesolver.vfcoor import VFCoor

from odevisualizationlib.evolution.stepper import RungaKutta4


class Evolution:
    def __init__(self, flow_equations: Any):
        from odevisualizationlib.modes import Evolution as VFEvolution
        self._evolution = VFEvolution(flow_equations._flow)

    def evolve(self, coordinates: VFCoor, stepper: Any=RungaKutta4(), start_t: float=0.0, end_t: Optional[float] = None, dt: float=0.01, n: Optional[int] = None, observer: Optional[Any] = None, max_steps: int=500, observe_every_ith_time_step: int=1, equidistant_time_observations: bool=True, step_size_type: str = "adaptive"):
        assert (n is None and end_t is not None) or (
                    n is not None and end_t is None), "Error: Note, either n or t1 needs to be find and only one of can be not None"
        if end_t is not None:
            self._evolution.evolve_const(stepper, coordinates.raw(), start_t, end_t, dt)
        # start_t, end_t, dt, n, observer, max_steps, observe_every_ith_time_step, equidistant_time_observations, step_size_type


        # stepper:
    # -   stepper, error stepper (constant dt)
    # -   controlled stepper, dense output stepper (variable dt)
        # from odevisualizationlib.evolve import evolve