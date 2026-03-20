# Defining geometry
import numpy as np
from typing import Callable

def make_geometry_mask(name: str, **params) -> Callable[[np.ndarray], np.ndarray]:

    if name == "cylinder":
        R = params["R"]
        zmin = params.get("z_min", 0.0)
        zmax = params.get("z_max", 1.0)

        def mask(r):
            x, y, z = r
            return (x**2 + y**2 < R**2) & (z >= zmin) & (z <= zmax)

        return mask

    elif name == "sphere":
        R = params["R"]

        def mask(r):
            return np.sum(r**2, axis=0) < R**2

        return mask

    else:
        raise ValueError(f"Unknown geometry: {name}")