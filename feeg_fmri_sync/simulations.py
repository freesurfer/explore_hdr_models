import numpy as np
import numpy.typing as npt

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from feeg_fmri_sync.constants import EEGData, fMRIData
from feeg_fmri_sync.models import HemodynamicModel
from feeg_fmri_sync.utils import (
    downsample_hdr_for_eeg,
    generate_descriptions_from_search_df,
    get_est_hemodynamic_response,
    get_hdr_for_eeg, 
    get_ratio_eeg_freq_to_fmri_freq,
    sum_hdr_for_eeg,
)

from feeg_fmri_sync.plotting import compare_est_fmri_with_actual

class ModelToFMRI(TypedDict):
    name: str
    fmri_data_options: Dict[str, fMRIData]
    fmri_to_plot: List[str]



def generate_downsampled_simulated_fmri(tr: float, r_fmri: float, eeg: EEGData, hrf: npt.NDArray, 
                                        noise: int, relu: bool = False, plot=False, title: Optional[str] = None,
                                        name: Optional[str] = None):
    hdr_for_eeg = get_hdr_for_eeg(eeg.data, hrf)
    fmri_hdr_for_eeg = downsample_hdr_for_eeg(r_fmri, hdr_for_eeg)
    noised_fmri = fmri_hdr_for_eeg + noise * np.random.default_rng().normal(size=fmri_hdr_for_eeg.shape)
    if relu:
        noised_fmri[noised_fmri < 0] = 0
    if plot:
        plot_kwargs = {}
        if title:
            plot_kwargs['est_fmri_label'] = f'Modeled fMRI: {name}'
            plot_kwargs['actual_fmri_label'] = 'Modeled fMRI without noise'
            plot_kwargs['title'] = title
        compare_est_fmri_with_actual(noised_fmri, fmri_hdr_for_eeg, eeg, tr, **plot_kwargs)
    return fMRIData(noised_fmri, tr, voxel_names=[name])


def generate_summed_simulated_fmri(
    tr: float, r_fmri: float, eeg: EEGData, hrf: npt.NDArray, noise: int, relu: bool = False, 
    plot: bool = False, title: Optional[str] = None, name: Optional[str] = None):
    hdr_for_eeg = get_hdr_for_eeg(eeg.data, hrf)
    fmri_hdr_for_eeg = sum_hdr_for_eeg(r_fmri, hdr_for_eeg)
    noised_fmri = fmri_hdr_for_eeg + noise * np.random.default_rng().normal(size=fmri_hdr_for_eeg.shape)
    if relu:
        noised_fmri[noised_fmri < 0] = 0
    if plot:
        plot_kwargs = {}
        if title:
            plot_kwargs['est_fmri_label'] = f'Modeled fMRI: {name}'
            plot_kwargs['actual_fmri_label'] = 'Modeled fMRI without noise'
            plot_kwargs['title'] = title
        compare_est_fmri_with_actual(noised_fmri, fmri_hdr_for_eeg, eeg, tr, **plot_kwargs)
    return fMRIData(noised_fmri, tr, voxel_names=[name])



def build_model_to_fmri(expected_delta: float, 
                        expected_tau: float, 
                        expected_alpha: float, 
                        num_trials: int,
                        num_trials_to_plot: int,
                        noise_range: npt.ArrayLike,
                        noises_to_plot: npt.ArrayLike,
                        plot_perfect_comparison: bool = True,
                        tr: float = 800, 
                        eeg_sample_freq: float = 20, 
                        hemodynamic_response_window: float = 30,
                        plot_generated_data: bool = False) -> Dict[HemodynamicModel, ModelToFMRI]:
    
    time_steps = np.arange(hemodynamic_response_window * eeg_sample_freq + 1) / eeg_sample_freq
    hrf = get_est_hemodynamic_response(time_steps, expected_delta, expected_tau, expected_alpha)
    r_fmri = get_ratio_eeg_freq_to_fmri_freq(eeg_sample_freq, tr)


    model_to_fmri = dict()

    for model, attrs in models_to_test.items():
        fmri_data_options = {}
        fmri_to_plot = []
        noise = 0
        name = f'{noise}noise_trial0'
        fmri_data_options[name] = attrs['fmri_data_generator'](
            tr, 
            r_fmri, 
            eeg_data_options['without_nans'],
            hrf,
            noise,
            plot=plot_generated_data,
            title=f'{model.__name__}_{name}',
            name = name,
        )
        if plot_perfect_comparison:
            fmri_to_plot.append(name)
        for trial in range(num_trials):
            for noise in noise_range:
                name = f'{noise}noise_trial{trial}'
                fmri_data_options[name] = attrs['fmri_data_generator'](
                    tr, 
                    r_fmri, 
                    eeg_data_options['without_nans'],
                    hrf,
                    noise,
                    plot=plot_generated_data if trial==0 else False,
                    title=f'{model.__name__}_{name}',
                    name=name
                )
                if noise in noises_to_plot and trial < num_trials_to_plot:
                    fmri_to_plot.append(name)
        model_to_fmri[model] = ModelToFMRI(
            name=attrs['name'],
            fmri_data_options=fmri_data_options,
            fmri_to_plot=fmri_to_plot
        )
    return model_to_fmri
