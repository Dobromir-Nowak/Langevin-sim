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


def const_initial_conditions_split(
    config: dict,
    r_const_1: np.ndarray,
    n_const_1: np.ndarray,
    r_const_2: np.ndarray,
    n_const_2: np.ndarray,
):
    N = config["N"]
    dim = config["dim"]
    if (
        dim != r_const_1.shape[0]
        or dim != n_const_1.shape[0]
        or dim != r_const_2.shape[0]
        or dim != n_const_2.shape[0]
    ):
        raise ValueError("Wrong array dimensions")
    if N % 2 != 0:
        raise ValueError("const_initial_conditions_split requires an even N")

    half_N = N // 2
    r_init = np.concatenate(
        (
            r_const_1[:, None] * np.ones(half_N),
            r_const_2[:, None] * np.ones(half_N),
        ),
        axis=1,
    )
    n_init = np.concatenate(
        (
            n_const_1[:, None] * np.ones(half_N),
            n_const_2[:, None] * np.ones(half_N),
        ),
        axis=1,
    )
    return r_init, n_init


# Approximation of exact integrals
# old approximation:
def f_new(I: np.ndarray, sin_psi:np.ndarray):  # up to order 3
    return (np.pi-2)*I/2 - I**2 * sin_psi / 3 + (3*np.pi - 4)/24 * I**3 * sin_psi**2


# new approximations:
def F1(I: np.ndarray, sin_psi:np.ndarray):  # phenomenological approximation of I*f1(x)x^{-1} valid for I*sin_psi in [0,10] interval
    x = I*sin_psi
    return I* (1+0.046*x)/(1+0.34*x)
def F2(I: np.ndarray, sin_psi:np.ndarray):  # phenomenological approximation of I*f2(x)x^{-1} valid for I*sin_psi in [0,10] interval
    x = I*sin_psi
    return I* (1+0.043*x)/(2/np.pi+0.267*x)

def F(I: np.ndarray, sin_psi: np.ndarray):
    return F2(I, sin_psi) - F1(I, sin_psi)
