import glob
import math
from typing import List

import numpy as np
import numpy.typing as npt
import os
import pandas as pd

from feeg_fmri_sync.constants import PROJECT_DIR, FMRI_DIR


def get_i_for_subj_and_run(subj: str, run: str, subj_and_run_list: List[str]):
    for i, subj_and_run in subj_and_run_list:
        if subj in subj_and_run and run in subj_and_run:
            return i
    raise ValueError(f'Cannot find subj {subj}, run {run} in {subj_and_run_list}')


def get_fmri_filepaths(root_dir, subject, hemi, run):
    files = []
    filename = f'res-{run.zfill(3)}.nii*'
    if hemi:
        for h in hemi:
            hemi_str = f'fsrest_{h}h_native'
            files.extend(glob.glob(os.path.join(root_dir, PROJECT_DIR, FMRI_DIR, subject, 'rest', hemi_str, 'res', filename)))
        return files
    # By default get all hemisphere
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
            ((time_steps[non_zero_mask] - delta)/tau)**alpha, 
            np.exp(-(time_steps[non_zero_mask] - delta)/tau)
        )
        # Scale so that max of continuous function is 1.
        # Peak will always be at (alpha.^alpha)*exp(-alpha)
        peak = alpha**alpha*math.exp(-alpha)
        hemodynamic_resp = hemodynamic_resp / peak
        return hemodynamic_resp


def get_hdr_for_eeg(eeg_data: npt.NDArray, hdr: npt.NDArray) -> npt.NDArray:
    return np.convolve(eeg_data, hdr, mode='full')[:eeg_data.shape[0]]


def get_ratio_eeg_freq_to_fmri_freq(eeg_freq: float, fmri_freq: float) -> float:
    return round(eeg_freq * fmri_freq / 1000)

def downsample_hdr_for_eeg(r_fmri: float, hdr_for_eeg: npt.NDArray) -> npt.NDArray:
    return hdr_for_eeg[::r_fmri]

def sum_hdr_for_eeg(r_fmri: float, hdr_for_eeg: npt.NDArray) -> npt.NDArray:
    hdr_to_chunk = hdr_for_eeg.copy()
    resize_tuple = (math.ceil(len(hdr_for_eeg) / r_fmri),r_fmri)
    hdr_to_chunk = np.resize(hdr_to_chunk, resize_tuple )
    return np.sum(hdr_to_chunk, axis=1)


def generate_descriptions_from_search_df(df, input_models = None):
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
        min_for_each_voxel.columns = description.iloc[:,voxel_indices].columns
        ret_desc = pd.concat((description, min_for_each_voxel))
        ret_desc.name = f'{model_name}'
        descriptions.append(ret_desc)
    return descriptions
