import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import time
import warnings

from typing import Dict, Any


from feeg_fmri_sync.constants import EEGData, fMRIData
from feeg_fmri_sync.models import HemodynamicModel


def get_suitable_range(start: int, end: int, num_points: int):
    """Return list with equal steps from [start, end] (including both points)"""
    step = (end - start) / (num_points - 1)
    return np.arange(start, end+step, step)


def search_voxel(models, delta_range, tau_range, alpha_range, get_all_data=False):
    warnings.warn("Soon to be deprecated! Use search_voxels")
    descriptions = []
    data = []
    for delta in delta_range:
        for tau in tau_range: 
            for alpha in alpha_range:
                scores = []
                for model in models.values():
                    scores.append(model.score(delta, tau, alpha))
                data.append([delta, tau, alpha]+scores)
    df = pd.DataFrame(data, columns=['delta', 'tau', 'alpha'] + [k for k in models.keys()])
    for model_name in models.keys():
        description = df[model_name].describe()
        description.name = f'{model_name}_{voxel}'
        for column_name in ['delta', 'tau', 'alpha']:
            min_val = df[df[model_name] == df[model_name].min()][column_name].iloc[0]
            description[column_name] = min_val
        descriptions.append(description)
    if get_all_data:
        return df
    return descriptions


def search_voxels(models, delta_range, tau_range, alpha_range):
    descriptions = []
    data = []
    model_names = np.array([k for k in models.keys()])
    all_voxel_names = [name for name in models[model_names[0]].fmri.voxel_names]
    for model_name, model in models.items():
        voxel_names = [name for name in model.fmri.voxel_names]
        if all_voxel_names != voxel_names:
            raise ValueError(f"All models must share voxel names in the same order to search together. {model_name} has different values from {model_names[0]}")
    if len(models) > 1:
        model_names = model_names.reshape((len(models), 1))
    for delta in delta_range: 
        print(f"Scoring delta={delta}")
        for tau in tau_range: 
            for alpha in alpha_range: 
                scores = []
                for model in models.values():
                    scores.append(model.score(delta, tau, alpha))
                vars = np.squeeze(np.ones((len(models), 3))*np.array([delta, tau, alpha]))
                scores = np.squeeze(np.array(scores))
                if len(models) > 1:
                    to_append = np.concatenate((vars, model_names, scores), axis=1)
                else:
                    to_append = np.concatenate((vars, model_names, scores))
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
        min_for_each_voxel.columns = description.iloc[:,voxel_indices].columns
        ret_desc = pd.concat((description, min_for_each_voxel))
        ret_desc.name = f'{model_name}'
        descriptions.append(ret_desc)    
    return descriptions, df


def run_simulated_search(model_to_fmri: Dict[HemodynamicModel, Dict[str, Any]], 
                         eeg_data_options: Dict[str, npt.NDArray],
                         tr: float,
                         n_tr_skipped_at_beginning: int,
                         hemodynamic_response_window: float,
                         plot: bool,
                         delta: npt.ArrayLike, 
                         tau: npt.ArrayLike,
                         alpha: npt.ArrayLike,
                         save_to: str = 'simulation.csv',
                         ideal_delta: float = 2.25,
                         ideal_tau: float = 1.25,
                         ideal_alpha: float = 2.) -> pd.DataFrame:
    models = {}
    for hemodynamic_model in model_to_fmri:
        fmri_names = [fmri_name for fmri_name in model_to_fmri[hemodynamic_model]['fmri_data_options'].keys()]
        fmri_data = np.array([fmri.data for fmri in model_to_fmri[hemodynamic_model]['fmri_data_options'].values()])
        fmri = fMRIData(data=fmri_data, TR=tr, voxel_names=fmri_names)
        for eeg_data_name, eeg_data in eeg_data_options.items():
            name = f'{model_to_fmri[hemodynamic_model]["name"]}_{eeg_data_name}'
            models[name] = hemodynamic_model(
                eeg=eeg_data,
                fmri=fmri,
                name=name,
                n_tr_skip_beg=n_tr_skipped_at_beginning,
                hemodynamic_response_window=hemodynamic_response_window,
                plot=plot
            )
            models[name].set_plot_voxels(model_to_fmri[hemodynamic_model]['fmri_to_plot']) 
    
    for model_name, model in models.items():
        residual_var = model.score(ideal_delta, ideal_tau, ideal_alpha)
        fmri_names = np.array(model.fmri.voxel_names)
        if (fmri_names == None).any():
            print(f'Residual variance was {residual_var}')
        else:
            noise_levels = np.char.replace(fmri_names, 'perfect', '0noise_trail0')
            noise_levels = np.char.partition(noise_levels, sep='noise')
            res_var_df = pd.DataFrame(zip(noise_levels[:,0].astype(int), residual_var), columns=['Noise', 'res_var'])
            res_var_by_noise = res_var_df.groupby('Noise')
            _, axs = plt.subplots()
            axs.set_title(model_name)
            axs.set_ylabel('Residual Variance')
            axs.set_xlabel('Noise')
            axs = res_var_by_noise.boxplot(column='res_var', subplots=False, rot=45, ax=axs)
            labels = res_var_by_noise.count()
            labels = [f'{noise}, N={n.item()}' for noise, n in zip(labels.index, labels.values)]
            plt.setp(axs, xticklabels=labels)
            plt.show()
        model.plot = False
    
    descriptions, df = search_voxels(models, delta, tau, alpha)
    with open(save_to, 'w') as f:
        pd.DataFrame(df).to_csv(f)
    for model_name, description in zip(models.keys(), descriptions):
        with open(f'{model_name}_summary_{save_to}', 'w') as f:
            pd.DataFrame(description).transpose().to_csv(f)
    return df
