import numpy as np
import numpy.typing as npt

import pandas as pd
import time

from typing import Dict, List, Type, Optional, Generator, Tuple, Iterable

from feeg_fmri_sync import CanonicalHemodynamicModel
from feeg_fmri_sync.constants import EEGData, fMRIData, STATISTICS
from feeg_fmri_sync.simulations import ModelToFMRI, generate_eeg_data


def get_suitable_range(start: int, end: int, num_points: int):
    """Return list with equal steps from [start, end] (including both points)"""
    step = (end - start) / (num_points - 1)
    return np.arange(start, end + step, step)


def build_models(model_to_fmri: Dict[Type[CanonicalHemodynamicModel], ModelToFMRI], eeg_data_options: List[str],
                 tr: float = 800, eeg_sample_freq: float = 20, eeg_data_by_name: Optional[Dict[str, EEGData]] = None,
                 **model_kwargs) -> Dict[str, CanonicalHemodynamicModel]:
    """
    Build models used in search_voxels
    """
    if not eeg_data_by_name:
        eeg_data_by_name = generate_eeg_data(eeg_data_options, eeg_sample_freq)
    models = {}
    for hemodynamic_model in model_to_fmri:
        fmri_names = [fmri_name for fmri_name in model_to_fmri[hemodynamic_model]['fmri_data_options'].keys()]
        fmri_data = np.array([fmri.data for fmri in model_to_fmri[hemodynamic_model]['fmri_data_options'].values()])
        fmri = fMRIData(data=fmri_data, TR=tr, voxel_names=fmri_names)
        for eeg_data_name, eeg_data in eeg_data_by_name.items():
            name = f'{model_to_fmri[hemodynamic_model]["name"]}_{eeg_data_name}'
            models[name] = hemodynamic_model(
                eeg=eeg_data,
                fmri=fmri,
                name=name,
                **model_kwargs
            )
            models[name].set_plot_voxels(model_to_fmri[hemodynamic_model]['fmri_to_plot'])
    return models


def search_voxels(models, delta_range, tau_range, alpha_range, verbose=True):
    descriptions = []
    data = []
    model_names = np.array([k for k in models.keys()])
    all_voxel_names = [name for name in models[model_names[0]].fmri.voxel_names]
    for model_name, model in models.items():
        voxel_names = [name for name in model.fmri.voxel_names]
        if all_voxel_names != voxel_names:
            raise ValueError(f"All models must share voxel names in the same order to search together. {model_name} "
                             f"has different values from {model_names[0]}")
    if len(models) > 1:
        model_names = model_names.reshape((len(models), 1))
    tstart = time.time()
    for i, delta in enumerate(delta_range):
        tend = time.time()
        if verbose and i > 0:
            print(f'Delta: {delta:.5f} ({i / len(delta_range) * 100:.2f}%). '
                  f'Last tau/alpha search took {tend - tstart:.2f} seconds')
        elif verbose:
            print(f'Delta: {delta:.5f} ({i / len(delta_range) * 100:.2f}%).')
        tstart = time.time()
        for tau in tau_range:
            for alpha in alpha_range:
                scores = []
                for model in models.values():
                    scores.append(model.score(delta, tau, alpha))
                search_vars = np.squeeze(np.ones((len(models), 3)) * np.array([delta, tau, alpha]))
                scores = np.squeeze(np.array(scores))
                if len(models) > 1:
                    to_append = np.concatenate((search_vars, model_names, scores), axis=1)
                else:
                    to_append = np.concatenate((search_vars, model_names, scores))
                data.append(to_append)

    data = np.vstack(data)
    df = pd.DataFrame(data, columns=['delta', 'tau', 'alpha', 'model_name'] + all_voxel_names)
    for model_name, model in models.items():
        df_for_model = df[df['model_name'] == model_name]
        voxel_names = [name for name in model.fmri.voxel_names]
        df_voxels = df_for_model[voxel_names].astype(float)
        description = df_voxels.describe()
        vdata = df_voxels.to_numpy()
        var_indices, voxel_indices = np.nonzero(vdata == vdata.min(axis=0))
        min_for_each_voxel = df_for_model.iloc[var_indices, :][['delta', 'tau', 'alpha']].transpose()
        min_for_each_voxel.columns = description.iloc[:, voxel_indices].columns
        ret_desc = pd.concat((description, min_for_each_voxel))
        ret_desc.name = f'{model_name}'
        descriptions.append(ret_desc)
    return descriptions, df.apply(pd.to_numeric, errors='ignore')


def search_voxels_in_depth(models: Dict[str, CanonicalHemodynamicModel], delta_range: npt.ArrayLike,
                           tau_range: npt.ArrayLike, alpha_range: npt.ArrayLike, verbose: bool = True,
                           get_significance: bool = True) -> pd.DataFrame:
    model_names = np.array([k for k in models.keys()])
    all_voxel_names = [name for name in models[model_names[0]].fmri.voxel_names]
    for model_name, model in models.items():
        voxel_names = [name for name in model.fmri.voxel_names]
        if all_voxel_names != voxel_names:
            raise ValueError(f"All models must share voxel names in the same order to search together. {model_name} "
                             f"has different values from {model_names[0]}")
    data = np.empty(
        (len(delta_range) * len(tau_range) * len(alpha_range) * len(models) * len(all_voxel_names), 11)
    )
    if len(models) > 1:
        model_names = model_names.reshape((len(models), 1))
    tstart = time.time()
    for d_i, delta in enumerate(delta_range):
        tend = time.time()
        if verbose and d_i > 0:
            print(f'Delta: {delta:.5f} ({d_i / len(delta_range) * 100:.2f}%). '
                  f'Last tau/alpha search took {tend - tstart:.2f} seconds')
        elif verbose:
            print(f'Delta: {delta:.5f} ({d_i / len(delta_range) * 100:.2f}%).')
        tstart = time.time()
        d_i_start = d_i * len(tau_range) * len(alpha_range) * len(models) * len(all_voxel_names)
        d_i_end = (d_i + 1) * len(tau_range) * len(alpha_range) * len(models) * len(all_voxel_names)
        data[d_i_start:d_i_end, 0] = delta
        for t_i, tau in enumerate(tau_range):
            t_i_start = d_i_start + t_i * len(alpha_range) * len(models) * len(all_voxel_names)
            t_i_end = d_i_start + (t_i + 1) * len(alpha_range) * len(models) * len(all_voxel_names)
            data[t_i_start:t_i_end, 1] = tau
            for a_i, alpha in enumerate(alpha_range):
                a_i_start = t_i_start + a_i * len(models) * len(all_voxel_names)
                a_i_end = t_i_start + (a_i + 1) * len(models) * len(all_voxel_names)
                data[a_i_start:a_i_end, 2] = alpha
                for m_i, model in enumerate(models.values()):
                    m_start = a_i_start + m_i * len(all_voxel_names)
                    m_end = a_i_start + (m_i + 1) * len(all_voxel_names)
                    data[m_start:m_end, 3] = m_i
                    data[m_start:m_end, 4] = np.arange(len(all_voxel_names))
                    info = model.score_detailed(delta, tau, alpha, get_significance=get_significance)
                    beta, _, residual_variance, degrees_of_freedom, r, pearsons_statistic, pearsons_pvalue = info
                    data[m_start:m_end, 5:7] = beta.T
                    data[m_start:m_end, 7] = residual_variance
                    data[m_start:m_end, 8] = r
                    data[m_start:m_end, 9] = pearsons_statistic
                    data[m_start:m_end, 10] = pearsons_pvalue

    df = pd.DataFrame(data, columns=['delta', 'tau', 'alpha', 'model_name', 'voxel_name'] + list(STATISTICS))
    for m_i, model_name in enumerate(model_names):
        df.loc[df['model_name'] == m_i, 'model_name'] = model_name[0]
    for v_i, voxel_name in enumerate(all_voxel_names):
        df.loc[df['voxel_name'] == v_i, 'voxel_name'] = voxel_name
    return df.apply(pd.to_numeric, errors='ignore')


def search_voxels_in_depth_without_df(
        models: Dict[str, CanonicalHemodynamicModel],
        delta_range: npt.ArrayLike,
        tau_range: npt.ArrayLike,
        alpha_range: npt.ArrayLike,
        verbose: bool = True,
        get_significance: bool = True) -> Tuple[npt.NDArray, List[str], List[Tuple[str, npt.ArrayLike]]]:
    model_names = np.array([k for k in models.keys()])
    all_voxel_names = [name for name in models[model_names[0]].fmri.voxel_names]
    for model_name, model in models.items():
        voxel_names = [name for name in model.fmri.voxel_names]
        if all_voxel_names != voxel_names:
            raise ValueError(f"All models must share voxel names in the same order to search together. {model_name} "
                             f"has different values from {model_names[0]}")
    data = np.empty(
        (len(delta_range), len(tau_range), len(alpha_range), len(models), len(all_voxel_names), 6)
    )
    tstart = time.time()
    for d_i, delta in enumerate(delta_range):
        tend = time.time()
        if verbose and d_i > 0:
            print(f'Delta: {delta:.5f} ({d_i / len(delta_range) * 100:.2f}%). '
                  f'Last tau/alpha search took {tend - tstart:.2f} seconds')
        elif verbose:
            print(f'Delta: {delta:.5f} ({d_i / len(delta_range) * 100:.2f}%).')
        tstart = time.time()
        for t_i, tau in enumerate(tau_range):
            for a_i, alpha in enumerate(alpha_range):
                for m_i, model in enumerate(models.values()):
                    info = model.score_detailed(delta, tau, alpha, get_significance=get_significance)
                    beta, _, residual_variance, _, r, pearsons_statistic, pearsons_pvalue = info
                    data[d_i, t_i, a_i, m_i, :, 0:2] = beta.T
                    data[d_i, t_i, a_i, m_i, :, 2] = residual_variance
                    data[d_i, t_i, a_i, m_i, :, 3] = r
                    data[d_i, t_i, a_i, m_i, :, 4] = pearsons_statistic
                    data[d_i, t_i, a_i, m_i, :, 5] = pearsons_pvalue
    indexers = [('delta', delta_range), ('tau', tau_range), ('alpha', alpha_range), ('models', model_names),
                ('voxels', all_voxel_names)]
    return_variable_names = ['beta_0', 'beta', 'residual_variance', 'correlation_coefficient', 'pearsons_statistic',
                             'pearsons_pvalue']
    if np.any(np.isnan(data[:, :, :, :, :, 4:])):
        return data[:, :, :, :, :, :4], return_variable_names[:4], indexers
    else:
        return data, return_variable_names, indexers


def analyze_best_fit_models(descriptions: List[pd.DataFrame],
                            models: Dict[str, CanonicalHemodynamicModel],
                            out_dir: Optional[str] = None,
                            display_plot: bool = False,
                            variable_to_minimize: str = 'residual_variance') -> Generator[
    Tuple[str, str, float, float, float, npt.NDArray, npt.NDArray, npt.NDArray, float, npt.NDArray, npt.NDArray,
    npt.NDArray], None, None]:
    if variable_to_minimize != 'residual_variance':
        raise NotImplementedError()
    for (model_name, model), description in zip(models.items(), descriptions):
        old_display_plot = model.display_plot
        old_out_dir = model.save_plot_dir
        old_plot_voxels = model.plot_voxels
        model.set_save_plot_dir(out_dir)
        model.display_plot = display_plot
        for column in description.columns:
            delta = float(description.loc['delta', column])
            tau = float(description.loc['tau', column])
            alpha = float(description.loc['alpha', column])

            model.plot_voxels = [column]
            info = model.score_detailed(delta, tau, alpha, column=column, get_significance=True)
            beta, residual, residual_variance, degrees_of_freedom, r, pearsons_statistic, pearsons_pvalue = info
            if not np.isclose(residual_variance, float(description.loc['min', column])):
                raise RuntimeWarning(f'Second scoring did not yield expected residual variance. Expected '
                                     f'{float(description.loc["min", column])}. Got {residual_variance.item()}')
            yield (model_name, column, delta, tau, alpha, beta, residual, residual_variance, degrees_of_freedom, r,
                   pearsons_statistic, pearsons_pvalue)

        model.save_plot_dir = old_out_dir
        model.display_plot = old_display_plot
        model.plot_voxels = old_plot_voxels
