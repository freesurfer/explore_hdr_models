import glob
import math
import warnings
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import os
import pandas as pd

from feeg_fmri_sync.constants import PROJECT_DIR, FMRI_DIR, fMRIData


def get_i_for_subj_and_run(subj: str, run: str, subj_and_run_list: List[str]):
    for i, subj_and_run in enumerate(subj_and_run_list):
        if subj in subj_and_run and run in subj_and_run:
            return i
    raise ValueError(f'Cannot find subj {subj}, run {run} in {subj_and_run_list}')


def get_fmri_filepaths(root_dir, subject, hemi, run):
    files = []
    filename = f'res-{run.zfill(3)}.nii*'
    if hemi:
        for h in hemi:
            hemi_str = f'fsrest_{h}h_native'
            files.extend(glob.glob(os.path.join(root_dir, PROJECT_DIR, FMRI_DIR, subject, 'rest', hemi_str, 'res',
                                                filename)))
        return files
    # By default, get all hemispheres
    hemi_str = f'fsrest_*h_native'
    return glob.glob(os.path.join(root_dir, PROJECT_DIR, FMRI_DIR, subject, 'rest', hemi_str, 'res', filename))


def get_est_hemodynamic_response(time_steps: npt.NDArray, delta: float, tau: float,
                                 alpha: float) -> npt.NDArray:
    """
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0;

    The return value is scaled so that the continuous-time peak = 1.0,
    though the peak of the sampled waveform may not be 1.0.

    """
    # First set to 0 to avoid RuntimeWarning about invalid value encountered in power
    hemodynamic_resp = np.zeros(time_steps.size)
    non_zero_mask = time_steps >= delta
    hemodynamic_resp[non_zero_mask] = np.multiply(
        ((time_steps[non_zero_mask] - delta) / tau) ** alpha,
        np.exp(-(time_steps[non_zero_mask] - delta) / tau)
    )
    # Scale so that max of continuous function is 1.
    # Peak will always be at (alpha.^alpha)*exp(-alpha)
    peak = alpha ** alpha * math.exp(-alpha)
    hemodynamic_resp = hemodynamic_resp / peak
    return hemodynamic_resp


def fit_glm(est_fmri: fMRIData, actual_fmri: fMRIData) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, int]:
    """
    Fit this time course to raw fMRI to waveform with a GLM:
        beta = inv(X'*X)*X'*fmri
        yhat = X*beta
        residual = fmri-yat
        residual variance = std(residual)
    Same effect as doing the correlation but residual variance is a cost we want to minimize
    rather than a correlation to maximize
    """
    if not est_fmri.is_single_voxel():
        warnings.warn(f'Estimated fMRI is multiple voxels. Model has not been tested')
    x_nan = np.isnan(est_fmri.data)
    x_drop_nans = np.extract(~x_nan, est_fmri.data)
    y_nan = np.tile(x_nan, actual_fmri.get_n_voxels()).reshape(
        (actual_fmri.get_n_voxels(), est_fmri.get_n_trs()))
    # np.extract flattens the array
    y_drop_nans_t = np.extract(~y_nan, actual_fmri.data).reshape(
        (actual_fmri.get_n_voxels(), x_drop_nans.shape[0]))
    # ones not necessary here
    x_t = np.array([x_drop_nans, np.ones(x_drop_nans.shape[0])])
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_t, x_t.T)), x_t), y_drop_nans_t.T)
    y_hat = np.matmul(x_t.T, beta)
    residual = np.subtract(y_drop_nans_t.T, y_hat).T
    degrees_of_freedom = x_t.shape[1] - x_t.shape[0]
    residual_variance = np.sum(residual ** 2, axis=actual_fmri.get_tr_axis()) / degrees_of_freedom
    return beta, residual, residual_variance, degrees_of_freedom


def get_hdr_for_eeg(eeg_data: npt.NDArray, hdr: npt.NDArray) -> npt.NDArray:
    return np.convolve(eeg_data, hdr, mode='full')[:eeg_data.shape[0]]


def get_ratio_eeg_freq_to_fmri_freq(eeg_freq: float, fmri_freq: float) -> float:
    return round(eeg_freq * fmri_freq / 1000)


def downsample_hdr_for_eeg(r_fmri: float, hdr_for_eeg: npt.NDArray) -> npt.NDArray:
    return hdr_for_eeg[::r_fmri]


def sum_hdr_for_eeg(r_fmri: float, hdr_for_eeg: npt.NDArray) -> npt.NDArray:
    hdr_to_chunk = hdr_for_eeg.copy()
    resize_tuple = (math.ceil(len(hdr_for_eeg) / r_fmri), r_fmri)
    hdr_to_chunk = np.resize(hdr_to_chunk, resize_tuple)
    return np.sum(hdr_to_chunk, axis=1)


def generate_descriptions_from_search_df(df, input_models=None):
    descriptions = []
    for model_name in df['model_name'].unique():
        df_for_model = df[df['model_name'] == model_name]
        if input_models:
            if model_name not in input_models:
                raise ValueError(f"Input models do not include {model_name}, which is in \n{df['model_name']}")
            voxel_names = [name for name in input_models[model_name].fmri.voxel_names]
        else:
            voxel_names = df.columns.drop(['model_name', 'delta', 'tau', 'alpha'])
        df_voxels = df_for_model[voxel_names].astype(float)
        description = df_voxels.describe()
        vdata = df_voxels.to_numpy()
        var_indices, voxel_indices = np.nonzero(vdata == vdata.min(axis=0))
        min_for_each_voxel = df_for_model.iloc[var_indices, :][['delta', 'tau', 'alpha']].transpose()
        min_for_each_voxel.columns = description.iloc[:, voxel_indices].columns
        ret_desc = pd.concat((description, min_for_each_voxel))
        ret_desc.name = f'{model_name}'
        descriptions.append(ret_desc)
    return descriptions
