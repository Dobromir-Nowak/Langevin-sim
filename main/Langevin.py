import numpy as np
import matplotlib.pyplot as plt

class Langevin_sim:
    def __init__(
        self,
        config: dict,
        r0: np.ndarray | None = None,
        n0: np.ndarray | None = None,
        is_history: bool = True
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

        # default intial conditions
        if r0 is not None:
            r0 = np.zeros((self.dim, self.N))
        if n0 is not None: 
            n0 = np.zeros((self.dim, self.N))
            n0[0,:]= 1
        self.r0 = r0
        self.n0 = n0

        # --- state ---
        self.is_history = is_history
        self.reset_and_initialize_state()

    def reset_and_initialize_state(
        self,
        r_new: np.ndarray | None = None,
        n_new: np.ndarray | None = None   
    )->None:
        """Initialize / reset system state."""
        if r_new is not None:
            self.r0 = r_new
        if n_new is not None:
            self.n0 = n_new
        
        self.r =  self.r0  # positions
        self.n = self.n0   # orientations

        # definition and reset of history (optional)
        if self.is_history:
            self.r_history = np.zeros((self.Nt+1, self.dim, self.N))
            self.r_history[0,:] = self.r0

            self.n_history = np.zeros((self.Nt+1, self.dim, self.N))
            self.n_history[0,:] = self.n0



    # TODO continue implementation from here
    # ------------------------
    # Dynamics
    # ------------------------
    def step(self):
        """One time step update."""
        noise = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.N)

        self.theta += noise
        direction = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)

        self.x += self.v0 * direction * self.dt

    # ------------------------
    # Simulation loop
    # ------------------------
    def run(self, save_every=1):
        for t in range(self.n_steps):
            self.step()

            if t % save_every == 0:
                self.record()

        return self.get_results()

    # ------------------------
    # Utilities
    # ------------------------
    def record(self):
        self.x_history.append(self.x.copy())
        self.theta_history.append(self.theta.copy())

    def get_results(self):
        return {
            "x": np.array(self.x_history),
            "theta": np.array(self.theta_history),
        }