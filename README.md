How to use: 
1. Clone github repo.
2. Ensure packages from reqs.txt are installed.
3. Set project parent directory as current directory.
4. Run "pip install -e ."
5. Any of the scripts and notebooks can now be run.

Jupyter notebooks (see folder "notebooks") can be run without the above setup.




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

