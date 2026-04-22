import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, F, const_initial_conditions
from langevin_sim.utils.other import load_config
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")


f_fn = F
I_fn = I_identity


file_name = "free"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

rm = ResultsManager(config_path=config_path, tag="free")

# Initial conditions
r_const = np.array([0,0,0],dtype=float)
n_const = np.array([1,0,0],dtype=float)
r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)




# Run simulation 
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, results_manager=rm)
# sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, results_manager=rm)
r, n = sim.run()
sim.plot_trajectories()
