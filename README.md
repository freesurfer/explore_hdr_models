# Fitting fEEG to fMRI
Package exploring how estimated fMRI data generated from 
Hemodynamic Models and an EEG spike train fits to observed fMRI.

* Does a grid search across the parameter space of the Hemodynamic Model
* Supports de-meaning (using zscore) of the estimated fMRI and/or the
  observed fMRI
* Supports filtering of the observed fMRI using a tunable Savgol Filter 
  or a tunable Gaussian filter.
* Supports missing data in the EEG spike train. 
* DOES NOT SUPPORT missing data in the observed fMRI!

This package provides 3 commands:
* `run_search_on_roi_gamma_model.py` Runs a parameter search on a fMRI timecourse
  that has been preprocessed and saved to a .mat file.
* `run_search_on_nii_gamma_model.py` runs a parameter search on an input nii
  file and eeg spike file. WARNING: this function has never been tested.
* `submit_from_config.py` Provides a wrapper to submit a set of search commands from 
  a configuration file.


## Installation
No package for this project exists at this time. 

You can create an environment where the scripts can run using poetry, 
which is good for local testing, or using conda, which is currently the
only supported batch submission environment.

### Poetry
1. Install Poetry on your system by following the instructions on 
   the official website: https://python-poetry.org/docs/#installation
2. Navigate to the root directory of your project using the terminal 
   or command prompt. 
3. Create a new virtual environment using Poetry by running the following 
   command: `poetry install`
   This command will read the `pyproject.toml` file and create a new 
   virtual environment with all the dependencies specified in the dependencies 
   and dev-dependencies sections. 
4. Once the virtual environment is created, you can activate it using the 
   following command: `poetry shell`
   This will activate the virtual environment and allow you to work within it.

You can now run your Python scripts, install additional dependencies using Poetry, 
and manage your environment using the `poetry` command-line tool. For example, 
you can add a new dependency to your project by running the following command:
`poetry add pandas`
This command will add the pandas package to your project and update the 
`pyproject.toml` file and lockfile (`poetry.lock`) accordingly.

To generate the requirements.txt file, run the following command:
`poetry export -f requirements.txt --without-hashes > requirements.txt`

Once you are done working in the virtual environment, you can exit it by 
running the following command: `exit`
This will deactivate the virtual environment and return you to your system's 
default Python environment.

Using Poetry to manage your environment can provide several benefits, 
such as better dependency resolution and reproducibility, easier 
management of virtual environments, and integration with other tools 
like Pytest and Flake8.

### Conda
1. Make sure you have Conda installed on your system. If not, you can 
   download and install Miniconda or Anaconda from the official Conda 
   website: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Open a terminal or command prompt.
3. Navigate to the directory where your `requirements.txt` file is located using the `cd` command. 
4. Create a new Conda environment with Python 3.9 by running the following command: `conda create -n myenv python=3.9`
   (Replace myenv with the desired name for your environment.)
5. Activate the newly created environment using the following command: `conda activate myenv`
6. Install the packages specified in the `requirements.txt` file using `pip` by 
   running the following command: `pip install -r requirements.txt`

You have now successfully created a Conda environment and installed the required 
packages. You can start using the environment by running your Python scripts or 
launching a Jupyter Notebook, depending on your project requirements.

Remember to activate the environment (`conda activate myenv`) every time you want 
to work within it, as it keeps your Python environment isolated and ensures that 
the installed packages are available for your project.


## Running a Search
This functionality is provided by `run_search_on_roi_gamma_model.py`.

Inputs:
* Sequence of EEG spikes - par file
* fMRI intensities - .nii or pre-processed .mat
* List of Hemodynamic Models
* Range of parameter values to search

Outputs:
* A few seconds of estimated hemodynamic responses plotted across the parameter search space.
* A pdf containing plots of the estimated fMRI timecourse generated at the default
  parameters (delta=2.25, tau=1.25, alpha=2):
  * Effect of de-meaning on the estimated fMRI timecourse, if applicable.
  * Comparison

Note: The ability to specify a list of hemodynamic models is not at its full functionality, since it does not 

Note: `run_search_on_nii_gamma_model.py` and `run_search_on_roi_gamma_model.py` have a regrettable amount of code 
      duplication, which has already caused

## Setup Searches Automatically
`submit_from_config.py` orchestrates building the searches and submitting them to a cluster.

Inputs:
* Optional flag
* Config file

Usage:
`python submit_from_config.py path/to/config/file.ini`

`example_search.ini` Provides an example config file that can be read using
`submit_from_config`.


## Supported Hemodynamic Models
Currently, all supported models are single


## Warnings/Known Bugs

If the output folders don't exist, the script will automatically
create them. However, this can cause a race condition if multiple 
scripts attempt to create the same output folders at the same time.
Rerunning the script will not generate another error.

## Acknowledgements
This README text was created with the help of ChatGPT4.