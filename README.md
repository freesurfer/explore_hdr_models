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

### Filtering Options

#### Savitzky-Golay Filter (Savgol Filter)
Savitzky-Golay (Savgol) filtering is a digital signal processing technique used to smooth or differentiate a signal.
It is particularly effective for removing noise or undesired fluctuations from a dataset. Savgol filtering operates 
by fitting a series of polynomial functions to subsets of the data and using these polynomial fits to estimate the 
smoothed or differentiated values.

The main parameters in Savgol filtering are the window length and the polynomial order. The window length refers 
to the number of adjacent data points considered for each polynomial fit. A larger window length incorporates more 
points in the fitting process and results in a smoother output, but it may also blur or remove small-scale features 
in the data. Conversely, a smaller window length captures more localized variations but may not effectively remove 
noise or provide a smooth result.

The polynomial order determines the complexity of the polynomial used to fit the data within each window. A higher 
polynomial order allows for more intricate fitting, which can capture more complex trends in the data. However, 
increasing the polynomial order can also lead to overfitting, where the filter amplifies noise or introduces 
artificial oscillations. A lower polynomial order provides a simpler fit, but it may fail to capture detailed 
variations in the signal.

To apply Savgol filtering, the algorithm slides the window along the signal, fitting a polynomial of the specified 
order to the points within the window. The polynomial coefficients are calculated using the least squares method,
minimizing the sum of squared residuals between the fitted curve and the actual data. The central point of the window 
is then replaced with the value obtained from evaluating the polynomial at that point. This process is repeated for 
all points in the signal.

It is important to note that Savgol filtering is a linear method and assumes that the signal is stationary within 
each window. If the signal contains abrupt changes or non-stationary behavior, the filtering may introduce artifacts 
or produce inaccurate results.

In summary, Savgol filtering is a technique for smoothing or differentiating signals using polynomial fits within 
moving windows. The window length and polynomial order are adjustable parameters that control the degree of smoothing 
and the ability to capture signal features. Care should be taken in selecting appropriate values to achieve the desired 
balance between noise removal and preserving important details in the data.

##### Good Starting Parameters
When applying Savgol filtering to fMRI data, the choice of filter parameters depends on the specific characteristics 
of the data and the goals of the analysis. However, there are some commonly used starting parameters that can provide 
a good starting point.

For fMRI data, a typical __window length__ is often set to be around __5 to 15__ time points. This value represents the 
number of consecutive time points that will be used to fit the polynomial within the sliding window. A larger window 
length will result in a smoother output, but it may also attenuate or remove certain transient features in the data. A 
smaller window length captures more localized variations but may not effectively remove noise.

Regarding the __polynomial order__, a value of __2__ is frequently used as a starting point for fMRI data. This means 
that a quadratic polynomial (2nd order) will be fitted to the data points within each window. A polynomial of higher 
order can capture more complex trends, but it also carries a higher risk of overfitting and amplifying noise. A lower 
polynomial order provides a simpler fit but may not capture subtle variations in the signal.

It's important to note that these starting parameters may vary depending on the specific characteristics of your 
fMRI data and the analysis objectives. It is recommended to experiment with different window lengths and polynomial 
orders to find the optimal balance between noise reduction and preservation of relevant signal features. Additionally, 
domain-specific knowledge and considerations, as well as the presence of any specific artifacts or noise sources in 
your fMRI data, should also be taken into account when selecting filter parameters.

#### Gaussian 1D Filter
Gaussian 1D filtering is a common technique used in image processing and signal processing to smooth data along 
one dimension. It applies a Gaussian kernel to the data, which is a bell-shaped curve defined by its mean (center) 
and standard deviation (spread). The kernel is convolved with the input signal to produce the filtered output.

The Gaussian kernel has the property of being symmetric and its shape is determined by the standard deviation. 
A higher standard deviation results in a wider kernel and more smoothing, while a lower standard deviation produces 
a narrower kernel and less smoothing. The mean of the Gaussian determines the center of the kernel.

The order of Gaussian filtering refers to the number of times the filter is applied to the data. Each subsequent 
application further smooths the data. Higher-order filtering can be useful in scenarios where more aggressive 
smoothing is required or to eliminate finer details from the data. However, it is important to note that higher-order 
filtering can also cause blurring and loss of important information, so the choice of order should be based on the 
specific requirements of the application.

The mode of Gaussian filtering refers to how the filter handles the boundary of the data. When applying the filter 
to the edges of the data, there is a region where the filter extends beyond the available data. The mode determines 
how this region is handled. Common modes include:

* "Valid" mode: In this mode, the filter is only applied to the valid region where the entire kernel fits within 
  the data. The output size is smaller than the input because the filter cannot be fully applied to the boundary 
  pixels.
* "Same" mode: In this mode, the filter is applied to the entire input signal, and the output size is the same as 
  the input size. The filter extends beyond the boundaries by assuming zero-padding outside the input data.
* "Wrap" mode: In this mode, the filter wraps around the data, treating the signal as if it were periodic. 
  The output size is the same as the input size, and the filter is applied to the entire signal, including the 
  boundary pixels.
* "Reflect" mode: In this mode, the filter is applied to the entire input signal, and the boundary pixels are 
  mirrored or reflected to handle the region beyond the boundaries. This mode helps to reduce artifacts that can 
  occur with other modes.

The choice of mode depends on the specific requirements of the application. "Valid" mode is often used when the 
filtered output should have the same size as the valid region of the input. "Same" mode is commonly used when the 
filtered output needs to have the same size as the input signal. "Wrap" and "reflect" modes are useful in situations
where the signal exhibits periodic behavior or to reduce boundary effects.

In summary, Gaussian 1D filtering involves convolving a Gaussian kernel with a one-dimensional signal to smooth 
the data. The order of the filter determines the level of smoothing, while the mode determines how the filter 
handles the boundary of the data.

##### Good Starting Parameters
When applying Gaussian 1D filtering to fMRI time courses, the choice of parameters depends on the specific 
characteristics of the data and the desired level of smoothing. Here are some considerations to guide 
parameter selection:
* Standard Deviation (σ): The standard deviation of the Gaussian kernel determines the width of the kernel and, 
  consequently, the amount of smoothing applied. A higher value of σ results in more smoothing, while a lower 
  value preserves finer details. The optimal σ depends on the characteristics of the fMRI data, such as the voxel 
  size and the underlying noise level. Common values for σ in fMRI analysis range from __2 to 6 seconds__. 
* Order: The order of the Gaussian filter determines the number of times the filter is applied consecutively. 
  Higher-order filtering provides more aggressive smoothing, but it can also introduce blurring and potentially 
  remove important temporal details. The choice of order should be based on the specific requirements of the 
  analysis. In many cases, a __first-order__ (single application) Gaussian filter is sufficient. 
* Mode: The mode of filtering determines how the filter handles the boundary of the fMRI time course. The 
  choice of mode depends on the specific requirements and characteristics of the data. "Same" mode is often used 
  to maintain the same length as the input time course, while "Valid" mode can be useful when removing edge effects 
  is a priority. "Reflect" and "Wrap" modes may be employed if there are periodic features in the time course or to 
  minimize boundary artifacts.

It is worth noting that the choice of parameters may require some experimentation and validation based on the specific 
characteristics of the fMRI data and the analysis goals. It is recommended to evaluate the impact of different 
parameter settings on the data and the subsequent analysis to ensure that the chosen parameters are appropriate for t
he particular study or task.

## Warnings/Known Bugs

If the output folders don't exist, the script will automatically
create them. However, this can cause a race condition if multiple 
scripts attempt to create the same output folders at the same time.
Rerunning the script will not generate another error.

## Acknowledgements
This README text was created with the help of ChatGPT4.