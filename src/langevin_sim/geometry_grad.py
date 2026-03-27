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
        phi = phi_stack[idx,np.arange(r.shape[1])] # phi_max.shape = (N,)

        grad = np.zeros_like(r)

        mask_radial = idx == 0
        grad[0, mask_radial] = x[mask_radial]/rho[mask_radial]
        grad[1, mask_radial] = y[mask_radial]/rho[mask_radial]

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
        phi = phi_stack[idx,np.arange(r.shape[1])] # phi_max.shape = (N,)

        return phi
    
    def grad_phi_all(self, r): # r.shape=(dim, N)
        return 

    def apply_template(self, r_old, r_new):
        phi_old, phi_new = self.phi_only(r_old), self.phi_only(r_new)

        if_outside = (phi_new>0)
        
        grad = self.grad_phi(r_old)
        # bounceback implementation
        r_new = ... # reference


    def apply(self, r_old, r_new):
            """
            r_old, r_new: shape (3, N)
            modifies r_new
            returns None
            """
            phi_old, phi_new = self.phi_only(r_old), self.phi_only(r_new)
            if_outside = (phi_new>0)

            it_max=5
            for it in range(it_max):
                if not np.any(if_outside):
                    break

                grad = self.grad_phi(r_new[:,if_outside])

                # bounceback implementation
                denominator = np.sum((r_new[:,if_outside]-r_old[:,if_outside])*grad, axis=0, keepdims=False)
                if np.any(np.abs(denominator) < 1e-8):
                    raise RuntimeError("Degenerate bounce: segment parallel to boundary")
                s = - phi_old[if_outside] / denominator

                delta1 = s*(r_new[:,if_outside]-r_old[:,if_outside])
                p = r_old[:,if_outside] + delta1 # intersection
                n = self.grad_phi(p) # normal to the boundary
                n /= np.linalg.norm(n, axis=0, keepdims=True)

                delta2 = (1-s)*(r_new[:,if_outside]-r_old[:,if_outside]) # remaining displacement
                delta2 = delta2 - 2*np.sum(delta2*n, axis=0)*n # reflecting the normal component of delta2

                # applying the bounceback
                r_new[:,if_outside] = r_old[:,if_outside] + delta1 + delta2

                # possible rerun
                phi_new = self.phi_only(r_new)
                if_outside = (phi_new>0)

                if it==it_max-1:
                    print(f"r_old = {r_old[:,if_outside]}")
                    print(f"p = {p}")
                    print(f"r_new = {r_new[:,if_outside]}")
                    print(f"phi_all = {self.grad_phi_all(r_new[:,if_outside])}")
                    raise RuntimeError(f"Bounce-back did not converge after {it_max} iterations."
                                       f"Trajectories unconverged: {np.sum(if_outside)}")




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