# Defining geometry
import numpy as np
from typing import Callable


class Cylinder3D:
    def __init__(self, R, zmin, zmax, bc="reflective"):
        self.R = R
        self.zmin = zmin
        self.zmax=zmax
        self.bc = bc
    
    # phi - signed distance function
    # positive outside, negative inside domain 
    def grad_phi(self, r): # r.shape=(dim, N)
        x, y, z = r[0,:], r[1,:], r[2,:]
        rho = np.sqrt(x**2 + y**2)
        phi_radial = rho - self.R
        phi_top = z - self.zmax
        phi_bottom = -z + self.zmin

        phi_stack = np.stack([phi_radial, phi_top, phi_bottom], axis=0)
        idx = np.argmax(phi_stack, axis=0)
        phi = phi_stack[idx] # phi_max.shape = (N,)


        grad = np.zeros_like(r)

        mask_radial = idx == 0
        grad[0, mask_radial] = x[mask_radial]/r[mask_radial]
        grad[1, mask_radial] = y[mask_radial]/r[mask_radial]

        mask_top = idx == 1
        grad[2, mask_top] = 1.

        mask_bottom = idx == 2
        grad[2, mask_bottom] = -1.

        return grad
    
    def phi_only(self, r): # r.shape=(dim, N)
        x, y, z = r[0,:], r[1,:], r[2,:]
        rho = np.sqrt(x**2 + y**2)
        phi_radial = rho - self.R
        phi_top = z - self.zmax
        phi_bottom = -z + self.zmin

        phi_stack = np.stack([phi_radial, phi_top, phi_bottom], axis=0)
        idx = np.argmax(phi_stack, axis=0)
        phi = phi_stack[idx] # phi_max.shape = (N,)

        return phi

    def apply(self, r_old, r_new):
        phi_old, phi_new = self.phi_only(r_old), self.phi_only(r_new)

        if_outside = (phi_new>0)
        
        grad = self.grad_phi(r_old)
        # bounceback implementation
        r = ... 
        return r



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