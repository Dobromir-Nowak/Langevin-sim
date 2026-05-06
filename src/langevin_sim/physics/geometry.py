# Defining geometry
import numpy as np
from typing import Callable

def reflect_orientation(n:np.ndarray, n_normal:np.ndarray):
    n_copy = np.copy(n)
    n_copy = n_copy - 2*n_normal * np.sum(n*n_normal,axis=0)
    return n_copy

def enforce_bounceback(r, bb_dim, domain_dims,n): # r.shape = (dim, N_samples)
    x_bb = r[bb_dim,:]
    Lx = domain_dims[bb_dim]

    # reflect orientation
    if_bb = np.logical_or(x_bb<=0,x_bb>=Lx)
    n[bb_dim,if_bb]*=-1

    # reflect position
    x_bb = np.abs(x_bb) # reflection about 0
    x_bb = Lx - np.abs(x_bb - Lx) # reflection about Lx; identity for x_bb in [0,Lx]; formula works for x in [0,2*Lx]
    r[bb_dim,:] = x_bb 

    if np.any(x_bb>Lx) or np.any(x_bb<0):
        print(x_bb[x_bb<0])
        print(x_bb[x_bb>Lx])
        raise ValueError(f"Spatial step too large in enforce_bounceback along axis {bb_dim}.")

def enforce_custom_bb(r_old, r_new, bb_dim, domain_dims,n,alpha): # r.shape = (dim, N_samples)
    x_bb = r_new[bb_dim,:]
    Lx = domain_dims[bb_dim]

    # reflect orientation
    if_bb_upper = x_bb>=Lx
    if_bb_lower = x_bb<=0
    if_any_bb = np.logical_or(if_bb_upper, if_bb_lower)
    n_proj = np.copy(n)
    n_proj[bb_dim,if_any_bb] = 0. # projection onto the plane

    norms = np.linalg.norm(n_proj, axis=0)      # normalizing the projected orientation
    zero_mask = norms == 0.0
    if np.any(zero_mask):
        raise ValueError(
            f"Zero-norm orientation vectors detected at indices "
            f"{np.where(zero_mask)[0]}")
    n_proj /= norms

    n[:,if_any_bb] = np.cos(alpha)*n_proj[:,if_any_bb] # changing in-plane components
    n[bb_dim, if_bb_upper] = -np.sin(alpha)
    n[bb_dim,if_bb_lower] = np.sin(alpha)


    # reflect position
    x_bb[if_any_bb]=r_old[bb_dim,if_any_bb]
    r_new[bb_dim,:] = x_bb 

    if np.any(x_bb>Lx) or np.any(x_bb<0):
        print(x_bb[x_bb<0])
        print(x_bb[x_bb>Lx])
        raise ValueError(f"Spatial step too large in enforce_bounceback along axis {bb_dim}.")

class NewCuboid:
    def __init__(self, config:dict, alpha: float):
        self.config = config
        self.Lx = config["Lx"]
        self.Ly = config["Ly"]
        self.Lz = config["Lz"]
        self.L = np.array([self.Lx, self.Ly, self.Lz])
        self.alpha = alpha

    def apply(self, r_old, r_new, n):
        """
        r_old, r_new, n: shape (3, N)
        modifies r_new and n
        returns None
        """
        for bb_dim in range(2):
            enforce_bounceback(r_new,bb_dim,self.L,n)
        bb_dim = 2
        enforce_custom_bb(r_old, r_new, bb_dim, self.L, n, self.alpha)


    def random_initial_conditions(self):
        
        dim = self.config["dim"]
        N = self.config["N"]

        # Initial positions
        rand = np.random.rand(dim,N) # np.random.rand(dim,N)*(1-2e-12)+1e-12

        r_init = self.L[:,None]*rand

        # Initial orientations
        n_rand = np.random.normal(loc=0,scale=1, size = (dim, N))
        n_init = n_rand/np.linalg.norm(n_rand, axis=0, keepdims=True)
        return r_init, n_init



class Cuboid:
    def __init__(self, config:dict):
        self.config = config
        self.Lx = config["Lx"]
        self.Ly = config["Ly"]
        self.Lz = config["Lz"]
        self.L = np.array([self.Lx, self.Ly, self.Lz])

    def apply(self, r_old, r_new, n):
        """
        r_old, r_new, n: shape (3, N)
        modifies r_new and n
        r_old is not needed here, but is kept for synctactic consistency
        returns None
        """
        for bb_dim in range(3):
            enforce_bounceback(r_new, bb_dim, self.L,n)


    def random_initial_conditions(self):
        
        dim = self.config["dim"]
        N = self.config["N"]

        # Initial positions
        rand = np.random.rand(dim,N) # np.random.rand(dim,N)*(1-2e-12)+1e-12

        r_init = self.L[:,None]*rand

        # Initial orientations
        n_rand = np.random.normal(loc=0,scale=1, size = (dim, N))
        n_init = n_rand/np.linalg.norm(n_rand, axis=0, keepdims=True)
        return r_init, n_init



class Cylinder3D:
    def __init__(
            self,
            config: dict,
            bc="reflective"):
        self.config = config
        self.R = config["R_cylinder"]
        self.zmin = config["zmin"]
        self.zmax = config["zmax"]
        self.bc = bc
    
    def phi_only(self, r): # r.shape=(dim, N)
        x, y, z = r[0,:], r[1,:], r[2,:]
        rho = np.sqrt(x**2 + y**2)
        phi_radial = rho - self.R
        phi_top = z - self.zmax
        phi_bottom = -z + self.zmin

        phi_stack = np.stack([phi_radial, phi_top, phi_bottom], axis=0)
        idx = np.argmax(phi_stack, axis=0)
        phi = phi_stack[idx,np.arange(phi_stack.shape[1])] # phi.shape = (N,)

        return phi

    def apply(self, r_old, r_new, n):
        """
        r_old, r_new, n: shape (3, N)
        modifies r_new and n
        returns None
        """
        current_start = np.array(r_old, dtype=float, copy=True)
        max_reflections = 8

        for _ in range(max_reflections):
            if_outside = self.phi_only(r_new) > 0
            if not np.any(if_outside):
                return

            start = current_start[:, if_outside]
            end = r_new[:, if_outside]
            n_selected = n[:, if_outside]
            hit, normal, hit_mask = self._first_boundary_hit(start, end, n_selected)
            n[:,if_outside] = n_selected
            if not np.all(hit_mask):
                # Fallback for numerically degenerate segments: keep particle at
                # the last known valid position instead of amplifying it.
                unresolved = np.where(if_outside)[0][~hit_mask]
                r_new[:, unresolved] = current_start[:, unresolved]
                if_outside[unresolved] = False

            if not np.any(hit_mask):
                continue

            active = np.where(if_outside)[0][hit_mask]
            remaining = end[:, hit_mask] - hit
            reflected = remaining - 2 * np.sum(remaining * normal, axis=0, keepdims=True) * normal

            r_new[:, active] = hit + reflected
            current_start[:, active] = hit
            
        # Final clamp for any particle still marginally outside after repeated
        # reflections, which can happen only from roundoff near edges/corners.
        if_outside = self.phi_only(r_new) > 0
        if np.any(if_outside):
            self._project_inside(r_new[:, if_outside])

    def _first_boundary_hit(self, r_old, r_new, n_selected):
        """Return first boundary hit along each segment from r_old to r_new."""
        npts = r_old.shape[1]
        dr = r_new - r_old
        s_best = np.full(npts, np.inf, dtype=float)
        normal = np.zeros_like(r_old)

        # Top plane z = zmax
        dz = dr[2]
        top_mask = (dz > 0) & (r_old[2] <= self.zmax) & (r_new[2] > self.zmax)
        if np.any(top_mask):
            s_top = (self.zmax - r_old[2, top_mask]) / dz[top_mask]
            valid = (s_top >= 0.0) & (s_top <= 1.0)
            idx = np.where(top_mask)[0][valid]
            s_best[idx] = s_top[valid]
            normal[2, idx] = 1.0
            n_selected[-1,idx] = -n_selected[-1,idx]

        # Bottom plane z = zmin
        bottom_mask = (dz < 0) & (r_old[2] >= self.zmin) & (r_new[2] < self.zmin)
        if np.any(bottom_mask):
            s_bottom = (self.zmin - r_old[2, bottom_mask]) / dz[bottom_mask]
            valid = (s_bottom >= 0.0) & (s_bottom <= 1.0)
            idx = np.where(bottom_mask)[0][valid]
            better = s_bottom[valid] < s_best[idx]
            idx = idx[better]
            s_best[idx] = s_bottom[valid][better]
            normal[:, idx] = 0.0
            normal[2, idx] = -1.0
            n_selected[-1,idx] = -n_selected[-1,idx]

        # Radial wall x^2 + y^2 = R^2
        dx = dr[0]
        dy = dr[1]
        x0 = r_old[0]
        y0 = r_old[1]
        a = dx**2 + dy**2
        b = 2.0 * (x0 * dx + y0 * dy)
        c = x0**2 + y0**2 - self.R**2
        radial_cross = (r_new[0]**2 + r_new[1]**2) > self.R**2
        radial_mask = radial_cross & (a > 1e-15)
        if np.any(radial_mask):
            disc = b[radial_mask]**2 - 4.0 * a[radial_mask] * c[radial_mask]
            disc = np.maximum(disc, 0.0)
            sqrt_disc = np.sqrt(disc)
            denom = 2.0 * a[radial_mask]
            s1 = (-b[radial_mask] - sqrt_disc) / denom
            s2 = (-b[radial_mask] + sqrt_disc) / denom
            s_candidates = np.stack([s1, s2], axis=0)
            s_candidates[(s_candidates < -1e-12) | (s_candidates > 1.0 + 1e-12)] = np.inf
            s_radial = np.min(s_candidates, axis=0)
            valid = np.isfinite(s_radial)
            idx = np.where(radial_mask)[0][valid]
            better = s_radial[valid] < s_best[idx]
            idx = idx[better]
            s_best[idx] = s_radial[valid][better]
            hit = r_old[:, idx] + s_best[idx] * dr[:, idx]
            rho = np.linalg.norm(hit[:2], axis=0)
            normal[:, idx] = 0.0
            normal[0, idx] = hit[0] / rho
            normal[1, idx] = hit[1] / rho
            n_reflected = reflect_orientation(n_selected[:,idx], normal[:,idx])
            n_selected[:,idx] = n_reflected # modify n with normal
        hit_mask = np.isfinite(s_best)
        hit = r_old[:, hit_mask] + s_best[hit_mask] * dr[:, hit_mask]
        normal_hit = normal[:, hit_mask]
        return hit, normal_hit, hit_mask

    def _project_inside(self, r):
        rho = np.sqrt(r[0]**2 + r[1]**2)
        radial_outside = rho > self.R
        if np.any(radial_outside):
            scale = self.R / rho[radial_outside]
            r[0, radial_outside] *= scale
            r[1, radial_outside] *= scale

        r[2] = np.clip(r[2], self.zmin, self.zmax)

    def random_initial_conditions(self):
        
        dim = self.config["dim"]
        N = self.config["N"]
        zmin = self.config["zmin"]
        zmax = self.config["zmax"]
        R = self.config["R_cylinder"]

        phi_rand = 2*np.pi*np.random.rand(N)
        radius_rand = R*np.sqrt(np.random.rand(N)) # important (change of variables)
        z_rand = zmin + (zmax-zmin)*np.random.rand(N)

        # Initial positions within the cylinder
        x = radius_rand*np.cos(phi_rand)
        y = radius_rand*np.sin(phi_rand)
        z = z_rand
        r_init = np.concatenate((x,y,z))
        r_init = r_init.reshape((dim,N))

        # Initial orientations within the cylinder
        n_rand = np.random.normal(loc=0,scale=1, size = (dim, N))
        n_init = n_rand/np.linalg.norm(n_rand, axis=0, keepdims=True)
        return r_init, n_init
