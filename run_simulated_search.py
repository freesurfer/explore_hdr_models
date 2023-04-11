import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from collections import defaultdict
from typing import Any, Dict, Tuple

from feeg_fmri_sync.constants import fMRIData
from feeg_fmri_sync.models import HemodynamicModel
from feeg_fmri_sync.plotting import (
    plot_all_search_results,
    plot_all_search_results_2d,
    plot_all_search_results_one_graph,
    plot_local_minima
)
from feeg_fmri_sync.search import get_suitable_range, run_simulated_search
from feeg_fmri_sync.utils import (
    get_est_hemodynamic_response,
    get_ratio_eeg_freq_to_fmri_freq,
)
from feeg_fmri_sync.vectorized_models import VectorizedHemodynamicModel, VectorizedSumEEGHemodynamicModel


from tests.helpers import (
    load_test_eeg_with_nans,
    load_test_eeg_without_nans,
    load_simulated_raw_fmri,
    generate_downsampled_simulated_fmri,
    generate_summed_simulated_fmri,
)

sample_freq = 20
tr = 800
n_tr_skipped_at_beginning = 0
hemodynamic_response_window = 30
save_to_filename = 'noise3_100.csv'
plot = False
recalculate = False

accurate_step_size = 41
step_size = 16

delta = get_suitable_range(1, 3, 41)
tau = get_suitable_range(0.75, 1.75, 41)
alpha = get_suitable_range(1.75, 2.25, step_size)

expected_delta = 2.25
expected_tau=1.25
expected_alpha=2

eeg_data_options = {
    'without_nans': load_test_eeg_without_nans(sample_frequency=sample_freq),
    #'with_nans': load_test_eeg_with_nans(sample_frequency=sample_freq),
}

time_steps = np.arange(hemodynamic_response_window*sample_freq + 1) / sample_freq
hrf = get_est_hemodynamic_response(time_steps, expected_delta, expected_tau, expected_alpha)
r_fmri = get_ratio_eeg_freq_to_fmri_freq(sample_freq, tr)
plot_data = False
model_to_fmri = {
    VectorizedHemodynamicModel: {
        'name': 'downsample',
        'fmri_data_options': {
            'perfect': generate_downsampled_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                0,
                plot=plot_data,
                title='perfect',
            ),
            '1noise': generate_downsampled_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                1,
                plot=plot_data,
                title='1noise',
            ),
            '2noise': generate_downsampled_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                2,
                plot=plot_data,
                title='2noise',
            ),
            'doug': load_simulated_raw_fmri(tr=tr),
            '5noise': generate_downsampled_simulated_fmri(
                tr,
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                5,
                plot=plot_data,
                title='5noise',
            ),
        }
    },
}
plot_skipped = False
skipped_model_to_fmri = {
    VectorizedSumEEGHemodynamicModel: {
        'name': 'sum',
        'fmri_data_options': {
            'perfect': generate_summed_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                0,
                plot=plot_skipped,
                title='perfect, summed',
            ),
            '1noise': generate_summed_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                1,
                plot=plot_skipped,
                title='1noise, summed'
            ),
            '2noise': generate_summed_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                2,
                plot=plot_skipped,
                title='2noise, summed'
            ),
            '3noise': generate_summed_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                3,
                plot=plot_skipped,
                title='3noise, summed'
            ),
            '5noise': generate_summed_simulated_fmri(
                tr,
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                5,
                plot=plot_skipped,
                title='5noise, summed'
            ),
        }
    }
}

noise3_model_to_fmri = {
    VectorizedHemodynamicModel: {
        'name': 'downsample',
        'fmri_data_options': {
            f'3noise{i}': generate_downsampled_simulated_fmri(
                tr, 
                r_fmri, 
                eeg_data_options['without_nans'],
                hrf,
                3,
                plot=plot_data,
                title='2noise',
            ) for i in range(100)
        }
    },
}


  

def plot_simulated_search_one_voxel(df, model_to_fmri, threed=False, one_graph=False):
    #TODO: search for local minimas
    for model in model_to_fmri.keys():
        columns = df.columns[np.char.startswith(df.columns.values.astype(str), model_to_fmri[model]['name'])]
        if one_graph:
            plot_all_search_results_one_graph(df[np.append(columns,['delta', 'tau', 'alpha'])])
            continue
        if threed:
            plot_all_search_results(df[np.append(columns,['delta', 'tau', 'alpha'])], separate_by='alpha')
            continue
        plot_all_search_results_2d(df[np.append(columns,['delta', 'tau', 'alpha'])], separate_by='alpha')


def plot_simulated_search(df, models_to_plot=None, fmri_to_plot: Tuple[str]=(), threed=False):
    #TODO: search for local minimas
    if not models_to_plot:
        models_to_plot = df['model_name'].unique() 
    for model in models_to_plot:
        df_to_plot = df[df['model_name'] == model].drop(columns='model_name')
        if fmri_to_plot:
            df_to_plot = df_to_plot[list(fmri_to_plot + ('alpha', 'delta', 'tau'))]
        if threed:
            plot_all_search_results(df_to_plot, separate_by='alpha')
            continue
        plot_all_search_results_2d(df_to_plot, separate_by='alpha')

if recalculate:
    df = run_simulated_search(
        noise3_model_to_fmri, #{**model_to_fmri, **skipped_model_to_fmri},
        eeg_data_options,
        tr,
        n_tr_skipped_at_beginning,
        hemodynamic_response_window,
        plot,
        delta, 
        tau,
        alpha,
        save_to=save_to_filename,
        ideal_delta=expected_delta,
        ideal_tau=expected_tau,
        ideal_alpha=expected_alpha
    )
else:
    with open(save_to_filename, 'r') as f:
        df = pd.read_csv(f)
        df = df.drop(columns='Unnamed: 0')

local_minima_for_model_column = plot_local_minima(df)





print(f"Expected minimal cost at delta={expected_delta:.2f}, tau={expected_tau:.2f}, alpha={expected_alpha:.2f}")
#plot_simulated_search(df, threed=False)
#plot_simulated_search(df, fmri_to_plot=('perfect',))