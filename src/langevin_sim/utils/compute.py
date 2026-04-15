from typing import Any

import matplotlib.pyplot as plt
import numpy as np



# Compute and print mean orientation
def compute_mean_n(n:np.ndarray)->np.ndarray:
    n_av = np.sum(n[-1,:], axis=-1)/n.shape[-1]
    print(f"Population averaged orientation n:\n {n_av}\n")
    return n_av

def compute_autocorr_n(n:np.ndarray)->np.ndarray:
    n_last = n[-1,:]
    autocorr = np.tensordot(n_last,n_last,axes=(1,1))/n.shape[-1]
    print(f"Population average of nn tensor:\n {autocorr}\n")
    return autocorr


# Gaussian beam light intensity field
def make_I_Gaussian_beam(config):
    sigma_beam = config["sigma_beam"]
    def I_Gaussian_beam(r): # r - positions to check field at:
        x, y = r[0,:], r[1,:]
        return np.exp(-(x**2 + y**2)/(2*sigma_beam**2))[None,:] # has to return shape (1, r.shape[1]), r.shape[1] = N
    return I_Gaussian_beam

# Unit light intensity field
def I_identity(r): # r - positions to check field at:
    return np.ones((1, r.shape[1]), dtype=float)   # has to return shape (1, r.shape[1]), r.shape[1] = N


def const_initial_conditions(config: dict, r_const: np.ndarray, n_const: np.ndarray):
    N=config["N"]
    dim=config["dim"]
    if dim!=r_const.shape[0] or dim!=n_const.shape[0]:
        raise ValueError("Wrong array dimensions")
    
    r_init = r_const[:,None]*np.ones(N)
    n_init = n_const[:,None]*np.ones(N)
    return r_init, n_init

