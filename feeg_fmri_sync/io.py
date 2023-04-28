import math
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy
import surfa as sf


def load_from_nii(
        nii_file: str,
        batch_number: Optional[int] = None,
        number_of_tasks: Optional[int] = None) -> Tuple[npt.NDArray, npt.NDArray]:
    fmri_data = sf.load_volume(nii_file)
    if batch_number:
        if not number_of_tasks:
            raise ValueError("Must specify number of tasks if batch_number is specified")
        n_voxels = fmri_data.shape[0]
        start = math.ceil(n_voxels / number_of_tasks) * batch_number
        end = min([math.ceil(n_voxels / number_of_tasks) * (batch_number + 1), n_voxels])
        return fmri_data[start:end, :, :, :].data.squeeze(), np.arange(start, end)
    return fmri_data[:, :, :, :].data.squeeze(), np.arange(fmri_data.shape[0])


def load_roi_from_mat(mat_file: str, subj_and_run_i: int) -> Tuple[npt.NDArray, npt.NDArray]:
    mat_data = scipy.io.loadmat(mat_file)
    if 'X' not in mat_data:
        raise ValueError(f'Expect variable "X" to be defined in {mat_file}')
    if 'subIndx' not in mat_data:
        raise ValueError(f'Expect variable "subIndx" to be defined in {mat_file}')
    fmri_data = mat_data['X']
    ind = mat_data['subIndx'].squeeze()
    return fmri_data[ind == subj_and_run_i, :].T, np.arange(1, fmri_data.shape[1]+1)
