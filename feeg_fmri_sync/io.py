import math
from typing import Optional, Tuple, List

import numpy as np
import numpy.typing as npt
import pandas as pd
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


def convert_df_to_mats(df: pd.DataFrame, mat_filename: str, columns_to_convert: Optional[List[str]] = None):
    if columns_to_convert is None:
        columns_to_convert = df.drop(columns=['delta', 'tau', 'alpha', 'model_name', 'voxel_name']).columns.to_list()
    for model_name in df['model_name'].unique():
        delta_range = df['delta'].unique()
        tau_range = df['tau'].unique()
        alpha_range = df['alpha'].unique()
        voxel_names = df['voxel_name'].unique()
        out_arrays = {column: np.empty((len(voxel_names), len(delta_range), len(tau_range), len(alpha_range)))
                      for column in columns_to_convert}
        for vi, voxel_name in enumerate(voxel_names):
            for di, delta in enumerate(delta_range):
                for ti, tau in enumerate(tau_range):
                    for ai, alpha in enumerate(alpha_range):
                        mask = np.all([
                            (df['model_name'] == model_name).tolist(),
                            (df['voxel_name'] == voxel_name).tolist(),
                            (df['delta'] == delta).tolist(),
                            (df['tau'] == tau).tolist(),
                            (df['alpha'] == alpha).tolist(),
                        ], axis=0)
                        for column in columns_to_convert:
                            out_arrays[column][vi, di, ti, ai] = df[mask][column].item()
        for column in columns_to_convert:
            scipy.io.savemat(f'{mat_filename}_{model_name}_{column}.mat', out_arrays[column])
