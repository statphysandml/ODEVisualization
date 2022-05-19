from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from odevisualizationlib import DevDat


class VFCoor:
    def __init__(self, devdat: any):
        self._devdat = devdat
    
    @classmethod
    def generate(cls, dim: int=3, N: int=1000, init_val: float=0.0) -> VFCoor:
        return cls(DevDat(dim, N, init_val))

    @classmethod
    def from_flattened_data(cls, data: List, dim: int):
        return cls(DevDat(data, dim))

    @classmethod
    def from_data(cls, data: List[List]=None):
        return cls(DevDat(data))

    def dim(self):
        return self._devdat.dim_size()

    def size(self):
        return self._devdat.n_elems()

    @property
    def shape(self):
        return self.size(), self.dim()

    @property
    def data(self):
        return self[:, :]

    def raw(self):
        return self._devdat

    def __str__(self):
        return str(self.data)

    @staticmethod
    def __fill(func: Callable[int, List, int], key: Union[int, slice], val: Union[int, List], n: int, sub: Optional[slice] = None):
        index_tuple = sub.indices(n)
        if isinstance(val, Iterable):
            assert len(val) == index_tuple[1] - index_tuple[0], "ValueError: could not broadcast input array of length " + str(len(val)) + " into slice of size (" + str(index_tuple[1] - index_tuple[0]) + ", )"
            func(key, val, index_tuple[0])
        else:
            func(key, [val] * (index_tuple[1] - index_tuple[0]), index_tuple[0])

    def __setitem__(self, key: Union[int, tuple, slice], val: Union[int, List, List[List]]):
        if isinstance(key, tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                self[key[0]:key[0] + 1, key[1]:key[1] + 1] = val
            # key[0] == int and key[1] == slice(preferred!)
            elif isinstance(key[0], int):
                self.__fill(func=self._devdat.fill_dim, key=key[0], val=val, n=self.size(), sub=key[1])
            # key[1] == int and key[0] == slice
            elif isinstance(key[1], int):
                self.__fill(func=self._devdat.set_nth_element, key=key[1], val=val, n=self.dim(), sub=key[0])
            elif key[0] == slice(None, None, None) and key[1] != slice(None, None, None) and isinstance(key[1], slice) and isinstance(val, list) and not isinstance(val[0], list):
                elem_index_tuple = key[1].indices(self.size())
                assert len(val) == elem_index_tuple[1] - elem_index_tuple[0], "ValueError: could not broadcoast input array of length " + str(len(val)) + " into slice of size (" + str(elem_index_tuple[1] - elem_index_tuple[0]) + ", )"
                for dim in range(*key[0].indices(self.dim())):
                    self[dim, key[1]] = val
            elif key[1] == slice(None, None, None) and key[0] != slice(None, None, None) and isinstance(key[0], slice) and isinstance(val, list) and not isinstance(val[0], list):
                dim_index_tuple = key[0].indices(self.dim())
                assert len(val) == dim_index_tuple[1] - dim_index_tuple[0], "ValueError: could not broadcoast input array of length " + str(len(val)) + " into slice of size (" + str(dim_index_tuple[1] - dim_index_tuple[0]) + ", )"
                for idx, dim in enumerate(range(*dim_index_tuple)):
                    self[dim, key[1]] = val[idx]
            else:
                if isinstance(val, Iterable):
                    elem_index_tuple = key[1].indices(self.size())
                    dim_index_tuple = key[0].indices(self.dim())
                    if isinstance(val[0], Iterable):
                        assert len(val) == dim_index_tuple[1] - dim_index_tuple[0] and len(val[0]) == elem_index_tuple[1] - elem_index_tuple[0], "ValueError: could not broadcast input array of shape (" + str(len(val)) + ", " + str(len(val[0])) + ") into slice of size (" + str(dim_index_tuple[1] - dim_index_tuple[0]) + ", " + str(elem_index_tuple[1] - elem_index_tuple[0]) + ")"
                        for dim, val_ in zip(range(*dim_index_tuple), val):
                            self[dim, key[1]] = val_
                    else:
                        assert False, "ValueError: could not broadcast input array of shape (" + str(len(val)) + ", ) into slice of size (" + str(self.dim()) + ", " + str(elem_index_tuple[1] - elem_index_tuple[0]) + ")"
                else:
                    for dim in range(*key[0].indices(self.dim())):
                        self[dim, key[1]] = val
        else:
            self[key, :] = val

    def __getitem__(self, key: Union[int, tuple, slice]):
        if isinstance(key, tuple):
            if isinstance(key[0], int) and isinstance(key[1], int):
                return self[key[0]:key[0] + 1, key[1]:key[1] + 1][0][0]
            # key[0] == int and key[1] == slice(preferred!)
            elif isinstance(key[0], int):
                elem_index_tuple = key[1].indices(self.size())
                return self._devdat.data_in_dim(key[0], elem_index_tuple[0], elem_index_tuple[1])
            # key[1] == int and key[0] == slice
            elif isinstance(key[1], int):
                dim_index_tuple = key[0].indices(self.dim())
                return self._devdat.get_nth_element(key[1], dim_index_tuple[0], dim_index_tuple[1])
            else:
                data = []
                for dim in range(*key[0].indices(self.dim())):
                    data.append(self[dim, key[1]])
                return data
        else:
            return self[key, :]
        
    
    def fill_data_in_dim(self, dim, data):
        self._devdat.fill_dim(dim, data, 0)

    def data_in_dim(self, dim):
        return self._devdat.data_in_dim(dim, 0, self._devdat.n_elems())

    def write_to_file(self, rel_dir, filename):
        self._devdat.write_to_file(rel_dir, filename)
        