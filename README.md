# L2S: Learning to Shine

## Overview

**L2S (Learning to Shine)** is an optimization framework designed to identify optimal illumination protocols for controlling optical phase transitions in materials with broken symmetry. By leveraging a neuroevolutionary approach, L2S optimizes the temporal structure of optical fields to achieve desired optomechanical responses.

This repository provides example workflows using continuous-wave and pulsed illumination strategies, along with workflows for Langevin dynamics, displacive excitation, and three-dimensional potential energy surface simulations.

## Repository Structure

### `L2S_continuous_wave_example.ipynb`

Demonstrates the application of the L2S optimization framework using a **continuous-wave** illumination strategy.

* Provides a step-by-step example run for the optimization workflow.
* Uses data from `Input_files/` for the potential energy and polarizability terms in the equation of motion.

### `L2S_pulsed_example.ipynb`

Illustrates the L2S optimization framework using **pulsed illumination protocols** for phase-transition control.

* Provides an example workflow analogous to the continuous-wave case.
* Uses data from `Input_files/` for the potential energy and polarizability terms in the equation of motion.

### `Input_files/`

Contains the potential energy and polarizability data derived from first-principles calculations.

These files serve as inputs to the equation of motion and enable realistic modeling of optomechanical interactions. They are used by the example notebooks and related workflows.

### `Langevin_dynamics/`

Contains the Langevin dynamics workflow used for optimization and dynamical simulations.

This folder includes:

* `src/`: source modules used by the Langevin dynamics and genetic-algorithm workflow.
* `Input_files/`: Langevin-specific input files used by the dynamics and optimization scripts.
* `run_ga.py`: genetic-algorithm driver script.
* `python-sub-langevin/`: job-submission and helper scripts for running Langevin dynamics calculations.

### `Displacive_excitation/`

Contains the notebook workflow for displacive pulsed excitation simulations.

This workflow is used to model optically driven displacive dynamics under pulsed excitation protocols.

### `3D_PES/`

Contains the notebook workflow for continuous-wave dynamics on a three-dimensional potential energy surface.

This folder includes the local helper module:

* `bi_pes3d_dft.py`

which is used to construct the three-dimensional Bi potential energy surface used by the notebook.

### `Final_figs_plots/Final_figs/`

Contains the manuscript-ready plotting notebooks, scripts, and supporting data used to generate the final figures presented in the manuscript.

The folder is organized as:

| Folder       | Manuscript figure |
| ------------ | ----------------- |
| `Fig1_abcd/` | Figure 1a–d       |
| `Fig1_e/`    | Figure 1e         |
| `Fig2/`      | Figure 2          |
| `Fig3/`      | Figure 3          |

The input data required for reproducing the manuscript plotting workflows are included, with one exception: a `*.pkl` file used for plotting displacive excitation dynamics results is not included because its size exceeds 10 GB.

## How to Use

Follow the steps below to install dependencies and run the provided notebooks.

### 1. Clone the Repository

```bash
git clone https://github.com/Sraddha-Agrawal/L2S.git
cd L2S
```

### 2. Create and Activate a Virtual Environment

Using a virtual environment is recommended to manage dependencies and avoid conflicts.

On macOS/Linux:

```bash
python3 -m venv l2s_env
source l2s_env/bin/activate
```

On Windows Command Prompt:

```bash
python -m venv l2s_env
l2s_env\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the main Python dependencies:

```bash
pip install numpy scipy matplotlib pandas torch torchvision ase jupyter
```

Several scripts also use standard-library modules such as `pathlib`, `itertools`, `copy`, `random`, `time`, `pickle`, `os`, `sys`, `glob`, and `math`. These do not need to be installed separately.

### 4. Verify Installation

To confirm that the main required libraries are correctly installed, run:

```bash
python -c "
import torch
import numpy
import scipy
import matplotlib
import pandas
import ase
print('All dependencies are installed successfully!')
"
```

If no errors appear and you see the success message, the installation was successful.

### 5. Open Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

### 6. Run the Example Optimization Notebooks

For continuous-wave optimization:

```bash
jupyter notebook L2S_continuous_wave_example.ipynb
```

For pulsed illumination optimization:

```bash
jupyter notebook L2S_pulsed_example.ipynb
```

### 7. Run the Langevin Dynamics Workflow

The Langevin dynamics workflow is located in:

```text
Langevin_dynamics/
```

The main genetic-algorithm driver is:

```text
Langevin_dynamics/run_ga.py
```

The Langevin-specific input files are located in:

```text
Langevin_dynamics/Input_files/
```

The source modules are located in:

```text
Langevin_dynamics/src/
```

and job-submission/helper scripts are located in:

```text
Langevin_dynamics/python-sub-langevin/
```

### 8. Run the Displacive Excitation Workflow

For displacive excitation simulations:

```bash
jupyter notebook Displacive_excitation/displacive_pulsed.ipynb
```

### 9. Run the 3D PES Workflow

For continuous-wave dynamics on a three-dimensional potential energy surface:

```bash
jupyter notebook 3D_PES/3D_PES_CW.ipynb
```

The local helper module required by this notebook is included in:

```text
3D_PES/bi_pes3d_dft.py
```

### 10. Reproduce Manuscript Figure Plots

Navigate to the relevant figure folder under:

```text
Final_figs_plots/Final_figs/
```

For example, to reproduce Figure 1a–d plots:

```bash
jupyter notebook Final_figs_plots/Final_figs/Fig1_abcd/Fig1_plots.ipynb
```

The figure folders are organized as:

```text
Final_figs_plots/Final_figs/Fig1_abcd/
Final_figs_plots/Final_figs/Fig1_e/
Final_figs_plots/Final_figs/Fig2/
Final_figs_plots/Final_figs/Fig3/
```

### 11. Modify Input Data

The top-level `Input_files/` directory contains potential energy and polarizability data derived from first-principles calculations for the main example notebooks.

To use custom input data, replace or add files in `Input_files/`:

```bash
cp path/to/custom_data.dat Input_files/
```

Confirm that the new files are in place:

```bash
ls Input_files/
```

For Langevin dynamics calculations, use the Langevin-specific input directory:

```text
Langevin_dynamics/Input_files/
```

## Notes on Large Files

All input data required by the included workflows are available in this repository, except for a `*.pkl` file used for plotting displacive excitation dynamics results. That file is not included because it is larger than 10 GB.

The omitted `*.pkl` file is a large plotting data file and is not required for running the main optimization examples. It is only needed for reproducing the corresponding cached displacive excitation dynamics plot directly from the saved object.

## Citation

An earlier version of this work is available on arXiv:

https://arxiv.org/pdf/2511.03895

The manuscript is currently being updated. A final citation will be provided here once the updated version is available.

If you use this repository in your research, please cite the arXiv preprint and/or the final published version when available.

## Contact

For questions, feedback, or contributions, please reach out via:

* **Email**: [shraddhaagrawal015@gmail.com](mailto:shraddhaagrawal015@gmail.com)
* **GitHub Issues**: [Submit an issue here](https://github.com/Sraddha-Agrawal/L2S/issues)
