from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable


class FixedPointCriterion:
    def __init__(self):
        from odevisualizationlib.recursive_search import FixedPointCriterion as VFFixedPointCriterion
        self._criterion = VFFixedPointCriterion()
