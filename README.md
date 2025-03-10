# L2S: Learning to Shine -- Overview 
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

## Required Libraries  
To run the notebooks and reproduce the results, install the following dependencies:  

```bash
pip install torch torchvision numpy scipy matplotlib ase pathlib itertools copy random time pickle

## Citation  

The manuscript detailing this work is currently in preparation. A citation link will be provided here once it is published.  

If you use this repository in your research, please acknowledge it by citing the upcoming publication.  

## Contact  

For any questions, feedback, or contributions, please reach out via:  

- **Email**: [shraddhaagrawal015@gmail.com]  
- **GitHub Issues**: [Submit an issue here](https://github.com/Sraddha-Agrawal/L2S/issues)  


