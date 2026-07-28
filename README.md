## How to use
1. Clone github repo.
2. Ensure packages from reqs.txt are installed.
3. Set project parent directory as current directory.
4. Run "pip install -e ."
5. Any of the scripts and notebooks can now be run.

Jupyter notebooks (see folder `notebooks`) can be run without the above setup.

## Example visualizations


https://github.com/user-attachments/assets/3a903b2f-f78a-4e9e-aabb-1f389fb5c81d

<img width="2519" height="1079" alt="currents_theta_t=100,N=8000_page-0001" src="https://github.com/user-attachments/assets/f4106a28-46d1-480b-8a7a-3cb07456c6fb" />

<img width="50%" alt="trajectories0_page-0001" src="https://github.com/user-attachments/assets/354fd133-6c3c-45e0-971d-2f9a0ca9fc0f" />

<!-- <img width="1444" height="1536" alt="trajectories0_page-0001" src="https://github.com/user-attachments/assets/354fd133-6c3c-45e0-971d-2f9a0ca9fc0f" /> -->

## Branch status and archive policy

The currently developed version of this project is the `main` branch. This branch was formerly developed as `cuboid-PBC` and is now treated as the active codebase. New development, bug fixes, configuration changes, and documentation updates should target `main`.

Older development states are preserved under branches named `archive/*`. These branches are kept for reference and reproducibility of older experiments. They are not maintained against the current `main` branch API.

Important consequences:

* `archive/*` branches should be treated as self-contained snapshots.
* Archived branches work with their own corresponding `src/` layout, scripts, and configuration files.
* Backward compatibility between archived branches and the current `main` branch is not implemented.
* Scripts or config files from archived branches should not be expected to run with the current `main/src` without manual adaptation.
* Conversely, current `main` scripts and configs should not be expected to run against archived `src` versions.

In particular, `archive/main` preserves the former main-branch implementation. It contains the largest set of older scripts and configuration examples, but it is no longer the active development branch. These scripts are useful as historical references for previous simulation variants, not as guaranteed-compatible entry points for the current simulator.

