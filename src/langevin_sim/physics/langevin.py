import numpy as np

from typing import Callable, Optional
from tqdm import tqdm
from langevin_sim.plotting.plots import plot_trajectories as plot_trajectories_from_history
from langevin_sim.physics.geometry import Cylinder3D, Cuboid
from langevin_sim.io.results import ResultsManager


# from .utils import plot_trajectories as plot_trajectories_from_history

class Langevin_sim:
    def __init__(
        self,
        config: dict,
        I_fn: Callable[[np.ndarray],np.ndarray],
        f_fn: Callable[[np.ndarray, np.ndarray],np.ndarray],
        D_fn: Callable[[np.ndarray],np.ndarray] | None = None,
        r0: np.ndarray | None = None,
        n0: np.ndarray | None = None,
        geometry: Cylinder3D | Cuboid | None = None,
        results_manager: ResultsManager | None = None
    )->None:

        # constants
        self.D = config.get("D", 0.)
        self.D_r = config.get("D_r", 0.067)

        # params
        self.dim = config["dim"]
        self.v0 = config["v0"]
        self.w0 = config["w0"]
        self.vec_I= np.array(config["vec_I"])
        if self.vec_I.size != self.dim:
            raise ValueError(f"vec_I has size {self.vec_I.size}, expected {self.dim}")
        
        # simulation params
        self.dt = config["dt"]
        self.Nt = config["Nt"]
        self.N = config["N"]

        # callable functions (light intensity and angular velocity)
        self.f_fn = f_fn
        self.I_fn = I_fn
        self.D_fn = D_fn

        # Intial conditions
        # default
        if r0 is None:
            r0 = np.zeros((self.dim, self.N), dtype=float)
        if n0 is None:
            n0 = np.zeros((self.dim, self.N), dtype=float)
            n0[-1, :] = 1.0
        # or custom
        self.r0 = r0
        self.n0 = n0
        
        # Initializing state
        self.r = np.array(self.r0, dtype=float, copy=True)
        self.n = np.array(self.n0, dtype=float, copy=True)

        # --- geometry ---
        self.geometry = geometry

        # --- results manager ---
        self.results_manager = results_manager

    def allocate_history(
        self,
        save_every: int
    )->None:
        self.save_every = save_every
        n_save = self.Nt // save_every + 1

        self.r_history = np.zeros((n_save, self.dim, self.N))
        self.n_history = np.zeros((n_save, self.dim, self.N))

        self.r_history[0] = self.r
        self.n_history[0] = self.n

        self.history_index = 0

    def _normalize_orientations(self)->None:
        norms = np.linalg.norm(self.n, axis=0)
        zero_mask = norms == 0.0
        if np.any(zero_mask):
            raise ValueError(
                f"Zero-norm orientation vectors detected at indices "
                f"{np.where(zero_mask)[0]}")
        self.n /= norms

    # ------------------------
    # Dynamics
    # ------------------------

    def Omega(self,sin_psi):  # I, psi have to have the same shape
        I_vals = self.I_fn(self.r)
        if I_vals.shape != sin_psi.shape:
            raise ValueError(f"I_vals shape {I_vals.shape} != psi shape {sin_psi.shape}")
        return self.f_fn(I_vals, sin_psi)  # same shape as x -- (1, self.N)

    def step(self):
        """One time step update."""
        r_old = self.r.copy()
        n_old = self.n.copy()

        # Position update
        self.r += self.v0*self.dt*n_old # deterministic step
        if self.D != 0.:
            self.r += np.sqrt(2.*self.D*self.dt) * np.random.randn(self.dim, self.N)

        if self.geometry is not None:
            self.geometry.apply(r_old, self.r, self.n) # modifies self.r and self.n in-place if needed

        n_old = self.n.copy() # orientations after possible geometry reflection, before orientation update

        # Orientation update
        if self.w0 != 0.:
            vec_I = self.vec_I[:, None]
            dot_prod = np.sum(n_old*(-vec_I), axis=0, keepdims=True)
            sin_psi = np.sqrt(np.maximum(0.0, 1.0 - dot_prod**2))  # r,n -- (self.dim, self.N), sin_psi -- (1, self.N)
            omega_values = self.w0*self.Omega(sin_psi)
            dn_rot = omega_values*self.dt*(-vec_I -dot_prod*n_old)
            self.n += dn_rot

        if self.D_r != 0.:
            if self.D_fn is not None:
                I_vals = self.I_fn(r_old)
                D_r = self.D_r*self.D_fn(I_vals) # shape (1, N)
            else:
                D_r = self.D_r
            dn_noise = np.sqrt(2.*D_r*self.dt) * np.random.randn(self.dim, self.N)
            dn_noise -= self.n*np.sum(n_old*dn_noise, axis=0, keepdims=True) 
            self.n += dn_noise

        self._normalize_orientations()


    # ------------------------
    # Simulation loop
    # ------------------------
    # is_history - are intermediate states stored in ram
    # save - is history (or final state) saved to file
    def run(self, save_every=1, is_history: bool = True, save: bool = False):
        self.is_history = is_history

        if save_every <= 0:
            raise ValueError("save_every must be a positive integer")
        if self.is_history:
            self.allocate_history(save_every=save_every)

        for t in tqdm(range(1,self.Nt+1)):
            self.step()

            if self.is_history and t % save_every==0:
                self.record()
        if self.Nt % save_every != 0: # saving the final state if unsaved after the iteration loop
            self.record()
    
        results = self.get_results()
        if save and self.results_manager is not None: # save results with manager
            if self.is_history: 
                self.results_manager.save_npz("trajectories", **results)
            else:
                self.results_manager.save_npz("final_state", **results)
        return results

    # ------------------------
    # Utilities
    # ------------------------
    def record(self):
        if self.history_index + 1 >= self.r_history.shape[0]:
            raise RuntimeError("History buffer overflow (check save_every vs Nt)")

        self.history_index += 1
        self.r_history[self.history_index, :] = self.r
        self.n_history[self.history_index, :] = self.n

    def get_results(self):
        if self.is_history:
            end = self.history_index + 1
            return {
                "r": self.r_history[:end],
                "n": self.n_history[:end],
            }

        return {
            "r": self.r.copy(),
            "n": self.n.copy(),
        }

    def plot_trajectories(
        self,
        r_history: np.ndarray | None = None,
        n_show: int = 50,
        ax=None,
        title: str = "Trajectories (subset)",
        show: bool = True,
        **kwargs,
    ):
        """Plot a subset of trajectories from stored history or supplied results."""
        if r_history is None:
            if not self.is_history:
                raise ValueError(
                    "Trajectory history is not enabled. Pass r_history explicitly or "
                    "initialize the simulator with is_history=True."
                )
            r_history = self.r_history[: self.history_index + 1]

        fig, ax = plot_trajectories_from_history(
            r_history=r_history,
            n_show=n_show,
            ax=ax,
            title=title,
            show=show,
            **kwargs)
        if self.results_manager is not None:
            self.results_manager.save_plot(fig, "trajectories")
        return fig
