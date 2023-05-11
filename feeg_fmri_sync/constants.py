import numpy as np
import numpy.typing as npt
import warnings

from collections import namedtuple
from typing import Optional, Tuple


# Statistics generated for each point on the search space
# TODO: Optimally, this would be grabbed from the model automatically
STATISTICS = ('beta_0', 'beta', 'residual_variance', 'correlation_coefficient', 'pearsons_statistic', 'pearsons_pvalue')

# Keys to be grabbed from the configuration file
SEARCH_KEYS = ['search-types', 'delta-start', 'delta-end', 'delta-step', 'tau-start', 'tau-end', 'tau-step',
               'alpha-start', 'alpha-end', 'alpha-step', 'save-data-to-mat', 'get-significance']

# Keys to be passed into the generation of each Hemodynamic Response Model instance
#    also used to grab keys from the configuration file (after replacing each _ with -)
# TODO: Optimally, this would be grabbed from the models automatically
HEMODYNAMIC_MODEL_KEYS = ['de_mean_est_fmri', 'de_mean_input_fmri', 'hemodynamic_response_window',
                          # savgol filter params
                          'savgol_filter_window_length', 'savgol_filter_polyorder', 'deriv',  'delta', 'mode', 'cval',
                          # gaussian filter params
                          'gaussian_filter_sigma', 'order', 'mode', 'cval', 'truncate', 'radius']

# Default 'mid-range' parameter values to generate example plots for
PLOT_DELTA = 2.25
PLOT_TAU = 1.25
PLOT_ALPHA = 2


# Tuple to connect EEG spike train data and the sample frequency
EEGData = namedtuple('EEGData', ['data', 'sample_frequency'])


# Class to connect fMRI data, TR, voxel names. Provides abstraction for certain methods,
# TODO: use of this abstraction wasn't fully tested, so there are probably some places in the code where the default
#       representation is still hard-coded
class fMRIData:
    """
    data: fMRI data (can be actual or estimated)
    TR: TR fMRI data was acquired at
    voxel_names: Optional list of strings to provide more human-readable texts for plots

    Let:
        n=number of fMRI timepoints (TR milliseconds apart)
        v=number of voxels
    If data is a 1D array, v is assumed to be 1 and dimension is expected to be (n,)
    If data is a 2D array, dimension is expected to be (v, n)
    """

    def __init__(self, data: npt.NDArray, TR: float, voxel_names: Optional[npt.ArrayLike] = None):
        if len(data.shape) > 2:
            raise ValueError(f"fMRI data is only expected to have 2 dimensions at most. "
                             f"Shape of given data is {data.shape}")
        self.data = data
        self.TR = TR
        if type(voxel_names) != np.ndarray and not voxel_names:
            self.voxel_names = np.array([None for _ in range(self.get_n_voxels())])
        else:
            if len(voxel_names) != self.get_n_voxels():
                raise ValueError(
                    f"Voxel names ({len(voxel_names)}) does not match number of voxels {self.get_n_voxels()}")
            self.voxel_names = np.array(voxel_names)
            assert np.unique(self.voxel_names).size == self.voxel_names.size, "Voxel names repeat"

    def get_n_trs(self) -> int:
        return self.data.shape[self.get_tr_axis()]

    def get_n_voxels(self) -> int:
        if self.is_single_voxel():
            return 1
        return self.data.shape[self.get_voxel_axis()]

    def is_single_voxel(self) -> bool:
        return len(self.data.shape) < 2

    def get_voxel_axis(self) -> int:
        if self.is_single_voxel():
            warnings.warn('Asked to get voxel axis of single voxel fMRI')
        return 0

    def get_tr_axis(self) -> int:
        if self.is_single_voxel():
            return 0
        return 1

    def get_voxel_i(self, i) -> Tuple[str, npt.NDArray]:
        if self.is_single_voxel():
            warnings.warn(f"Asked to get voxel {i}, only one voxel available")
            return self.voxel_names[0], self.data
        # TODO: is it possible to return this dynamically?
        # WARNING: if axis changes, this will currently fail
        return self.voxel_names[i], self.data[i, :]

    def get_voxel_by_name(self, name) -> Tuple[npt.NDArray, npt.NDArray]:
        if self.is_single_voxel():
            if self.voxel_names != [name]:
                raise ValueError(f'{name} is not present in voxel_names: {self.voxel_names}')
            return np.zeros(1), self.data
        i = np.where(self.voxel_names == name)
        if len(i) != 1:
            raise ValueError(f'{name} present more than once in voxel_names: {self.voxel_names}')
        if i[0].size != 1:
            raise ValueError(f'{name} is not present in voxel_names: {self.voxel_names}')
        # TODO: is it possible to return this dynamically?
        # WARNING: if axis changes, this will currently fail
        return i, self.data[i[0].item(), :]
