# L2S: Learning to Shine 

## Overview
L2S (Learning to Shine) is an optimization framework designed to identify optimal illumination protocols for controlling optical phase transitions in materials with broken symmetry. By leveraging a neuroevolutionary approach, L2S optimizes the temporal structure of optical fields to achieve desired optomechanical responses. This repository provides example workflows using continuous wave and pulsed illumination strategies.  

## Repository Structure  

- **`L2S_continuous_wave_example.ipynb`**  
  - Demonstrates the application of the L2S optimization framework using a **continuous waveform** illumination strategy.  
  - Includes a step-by-step example run, guiding users through the optimization process.  
  - Uses data from `Input_files` for potential energy and polarizability in the equation of motion.  

- **`L2S_pulsed_example.ipynb`**  
  - Illustrates the optimization framework when using **pulsed signal protocols** for phase transition control.  
  - Similar to the continuous wave example, this notebook walks through an example run.  
  - Uses data from `Input_files` for potential energy and polarizability in the equation of motion.  

- **`Input_files/`**  
  - Contains essential **potential energy** and **polarizability** data, derived from first-principles calculations.  
  - This data serves as input to the **equation of motion**, enabling realistic modeling of optomechanical interactions.  
  - Used in both `L2S_continuous_wave_example.ipynb` and `L2S_pulsed_example.ipynb`.  

- **`Final_figs_plots/Final_figs/`**  
  - This directory holds the input datasets (located in the `Data` subfolder) and Jupyter Notebook scripts used to generate the four key figures presented in the manuscript.  
  - The scripts allow users to reproduce the plots and analyze the results presented in the study.
 
## How to Use  

Follow the steps below to install dependencies and run the provided notebooks:

### 1. Clone the Repository  

First, clone this repository to your local machine:  
```bash
git clone https://github.com/Sraddha-Agrawal/L2S.git
cd L2S
```

### 2. Create and Activate a Virtual Environment (Recommended) 

It is recommended to create a virtual environment to manage dependencies and avoid conflicts.

### On macOS/Linux:
```bash
python3 -m venv l2s_env
source l2s_env/bin/activate
```

### On Windows (Command Prompt):
```bash
python -m venv l2s_env
l2s_env\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib ase pathlib itertools copy random time pickle
```

### 4. Verify Installation

To confirm that all necessary libraries are correctly installed, run the following command:

```bash
python -c "
import torch
import numpy
import scipy
import matplotlib
import ase
print('All dependencies are installed successfully!')
"
```
If no errors appear and you see the success message, the installation was successful.

### 5. Open Jupyter Notebook

Launch Jupyter Notebook to run the optimization scripts:

```bash
jupyter notebook
```

### 6. Run the Desired Notebook

Open and execute the appropriate notebook based on the desired illumination protocol:

### For Continuous Wave Optimization:
```bash
jupyter notebook L2S_continuous_wave_example.ipynb
```
### For Pulsed Signal Optimization:
```bash
jupyter notebook L2S_pulsed_example.ipynb
```

### 7. Modify Input Data (Optional)

The `Input_files/` directory contains essential **potential energy** and **polarizability** data derived from first-principles calculations.

If you wish to modify the input data, follow these steps:

### Replace the default input files with custom data:
```bash
cp path/to/custom_data.dat Input_files/
```

Confirm that the new files are in place:
```bash
ls Input_files/
```

## Citation  

The manuscript detailing this work is currently in preparation. A citation link will be provided here once it is published.  

If you use this repository in your research, please acknowledge it by citing the upcoming publication.  

## Contact

For any questions, feedback, or contributions, please reach out via:  

- **Email**: [shraddhaagrawal015@gmail.com]  
- **GitHub Issues**: [Submit an issue here](https://github.com/Sraddha-Agrawal/L2S/issues)  


