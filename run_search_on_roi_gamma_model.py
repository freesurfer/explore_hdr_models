"""
Run grid search to find best fitting delta, tau, and alpha using the specified hemodynamic model.

By default, uses canonical hemodynamic model:
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0

Usage:
python run_search_on_roi_gamma_model.py \
    --par-file /autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par \
    --mat-file /autofs/space/ursa_004/users/HDRmodeling/ROItcs_fMRI/mat4HMMregWindu.mat \
    --sub-and-run-i 1
    -v
    --out-dir /autofs/space/ursa_004/users/HDRmodeling/HDRmodeling/DAN/s06_137-r1/
    --out-name DAN_s06_137_r1

"""
import argparse
import json

import numpy as np
import os
import pandas as pd
import scipy

from feeg_fmri_sync import SEARCH_TYPES
from feeg_fmri_sync.constants import PLOT_ALPHA, PLOT_DELTA, PLOT_TAU
from feeg_fmri_sync.io import load_roi_from_mat
from feeg_fmri_sync.constants import EEGData, fMRIData
from feeg_fmri_sync.plotting import (
    plot_all_search_results_2d_on_diff_colormaps,
    plot_eeg_hdr_across_delta_tau_alpha_range,
    save_plot
)
from feeg_fmri_sync.search import search_voxels, analyze_best_fit_models, search_voxels_in_depth_without_df

parser = argparse.ArgumentParser()

parser.add_argument('--get-significance', action='store_true')

# EEG info
parser.add_argument('--par-file', required=True, help=f'par file path')
parser.add_argument('--eeg-sample-frequency', type=int, default=20)

# fMRI file info
parser.add_argument('--mat-file', required=True, help=f'ROI mat file path (must define X and subIndx as variables)')
parser.add_argument('--sub-and-run-i', required=True, type=int, help='subIndx value in mat-file for eeg par file')
parser.add_argument('--tr', type=int, default=800)
parser.add_argument('--num-trs-skipped-at-beginning', type=int, default=1)

# Output info
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--out-dir', required=True, help=f'out file directory')
parser.add_argument('--out-name', required=True, help=f'Name for model')
parser.add_argument('--save-data-to-mat', action='store_true')

# Gamma Parameters to Search Over
parser.add_argument('--delta-start', type=float, default=1)
parser.add_argument('--delta-end', type=float, default=3, help='inclusive')
parser.add_argument('--delta-step', type=float, default=0.05)
parser.add_argument('--tau-start', type=float, default=0.75)
parser.add_argument('--tau-end', type=float, default=1.75, help='inclusive')
parser.add_argument('--tau-step', type=float, default=0.05)
parser.add_argument('--alpha-start', type=float, default=1.75)
parser.add_argument('--alpha-end', type=float, default=2.25, help='inclusive')
parser.add_argument('--alpha-step', type=float, default=0.05)


# Search Type
parser.add_argument('--search-type', default='classic_hemodynamic', choices=SEARCH_TYPES.keys())
parser.add_argument('--standardize', action='store_true')
parser.add_argument('--hemodynamic-response-window', type=float, default=30)
parser.add_argument('--savgol-filter-window-length', type=int, default=5)
parser.add_argument('--savgol-filter-polyorder', type=int, default=5)
parser.add_argument('--deriv', type=int)
parser.add_argument('--delta', type=float)
parser.add_argument('--mode', type=str)
parser.add_argument('--cval', type=float)

SEARCH_KWARG_NAMES = ['hemodynamic_response_window', 'savgol_filter_window_length', 'savgol_filter_polyorder', 'deriv',
                      'delta', 'mode', 'cval']

if __name__ == '__main__':
    args = parser.parse_args()

    # Load eeg data
    eeg_data = np.fromfile(args.par_file, sep='\n')
    eeg = EEGData(eeg_data, args.eeg_sample_frequency)

    # Load fmri data
    fmri_voxel_data, fmri_voxel_names = load_roi_from_mat(args.mat_file, args.sub_and_run_i)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    delta_range = np.arange(args.delta_start, args.delta_end + args.delta_step, step=args.delta_step)
    tau_range = np.arange(args.tau_start, args.tau_end + args.tau_step, step=args.tau_step)
    alpha_range = np.arange(args.alpha_start, args.alpha_end + args.alpha_step, step=args.alpha_step)

    search_kwargs = {kn: kv for kn, kv in vars(args).items() if kn in SEARCH_KWARG_NAMES}
    models = {f'{args.search_type}_{args.out_name}': SEARCH_TYPES[args.search_type]['model'](
        eeg,
        fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
        args.out_name,
        args.num_trs_skipped_at_beginning,
        display_plot=False,
        standardize=args.standardize,
        **search_kwargs
    )}
    # Create a model to plot actual fMRI vs estimated
    model_for_plotting = SEARCH_TYPES[args.search_type]['model'](
        eeg,
        fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
        args.out_name,
        args.num_trs_skipped_at_beginning,
        display_plot=False,
        save_plot_dir=args.out_dir,
        standardize=args.standardize,
        **search_kwargs
    )
    # Save plot of how the variables change relating to each other across the search space
    save_plot(
        os.path.join(args.out_dir, f'{args.out_name}_across_search_space'),
        plot_eeg_hdr_across_delta_tau_alpha_range,
        eeg,
        args.hemodynamic_response_window,
        args.tr,
        delta_range,
        tau_range,
        alpha_range,
    )
    model_for_plotting.set_plot_voxels(fmri_voxel_names)
    model_for_plotting.score(PLOT_DELTA, PLOT_TAU, PLOT_ALPHA)

    if args.save_data_to_mat:
        data, variable_names, indexers = search_voxels_in_depth_without_df(models, delta_range, tau_range, alpha_range,
                                                                           args.verbose, args.get_significance)

        for v_i, variable_name in enumerate(variable_names):
            out_name = f'{variable_name}_search_on_' \
                       f'{os.path.basename(args.mat_file).split(".")[0]}_sub{args.sub_and_run_i}.mat'
            if args.verbose:
                print(f'Writing {variable_name} search to {out_name}')
            print(data.shape)
            print(len(data.shape) - 1)
            print(v_i, variable_name)
            scipy.io.savemat(
                os.path.join(args.out_dir, out_name),
                {variable_name: data.take(v_i, axis=(len(data.shape) - 1))}
            )

        for i, (indexer_name, indexer_values) in enumerate(indexers):
            out_name = f'{i}_key_{indexer_name}_for_{os.path.basename(args.mat_file).split(".")[0]}' \
                       f'_sub{args.sub_and_run_i}.csv'
            np.savetxt(os.path.join(args.out_dir, out_name), indexer_values, delimiter=",", fmt="%s")

    else:
        descriptions, df = search_voxels(models, delta_range, tau_range, alpha_range, args.verbose)

        for data_packet in analyze_best_fit_models(descriptions, models, args.out_dir):
            (model_name, column, delta, tau, alpha, beta, _, residual_variance, dof, r,
             pearsons_statistic, pearsons_pvalue) = data_packet
            ret_dict = {
                'model_name': model_name,
                'column': str(column),
                'delta': float(delta),
                'tau': float(tau),
                'alpha': float(alpha),
                'beta': {
                    'beta_0': float(beta[0][0]),
                    'beta': float(beta[1][0])
                },
                'residual_variance': float(residual_variance),
                'degrees_of_freedom': int(dof),
                'correlation_coefficient': float(r),
                'pearsons_statistic': float(pearsons_statistic),
                'pearsons_pvalue': float(pearsons_pvalue)
            }

            out_name = f'{model_name}_best_fit' \
                       f'{os.path.basename(args.mat_file).split(".")[0]}_sub-{args.sub_and_run_i}_column-{column}.json'
            if args.verbose:
                print(f'Writing Best fit model for model {model_name}, {column} to {out_name}')
            with open(os.path.join(args.out_dir, out_name), 'w') as f:
                json.dump(ret_dict, f)

        for model_name, description in zip(df['model_name'].unique(), descriptions):
            parameters_chosen_by_search = []
            df_to_plot = df[df['model_name'] == model_name].drop(columns='model_name').astype(float)
            if args.verbose:
                print(f'Plotting search results...')
            save_plot(
                os.path.join(args.out_dir, f'{model_name}_cost_heat_map'),
                plot_all_search_results_2d_on_diff_colormaps,
                df_to_plot,
                verbose=False,
            )
            out_name = f'{model_name}_search_summary_on_' \
                       f'{os.path.basename(args.mat_file).split(".")[0]}_sub{args.sub_and_run_i}.csv'
            if args.verbose:
                print(f'Writing search summary to {out_name}')
            with open(os.path.join(args.out_dir, out_name), 'w') as f:
                pd.DataFrame(description).to_csv(f)
