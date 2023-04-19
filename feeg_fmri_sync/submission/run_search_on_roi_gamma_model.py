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
import numpy as np
import os
import pandas as pd

from feeg_fmri_sync import SEARCH_TYPES
from feeg_fmri_sync.constants import PLOT_ALPHA, PLOT_DELTA, PLOT_TAU
from feeg_fmri_sync.io import load_roi_from_mat
from feeg_fmri_sync.models import EEGData, fMRIData
from feeg_fmri_sync.plotting import (
    plot_all_search_results_2d_on_diff_colormaps,
    plot_eeg_hdr_across_delta_tau_alpha_range,
    save_plot
)
from feeg_fmri_sync.search import search_voxels


parser = argparse.ArgumentParser()

# EEG file info
parser.add_argument('--par-file', required=True, help=f'par file path')

# fMRI file info
parser.add_argument('--mat-file', required=True, help=f'ROI mat file path (must define X and subIndx as variables)')
parser.add_argument('--sub-and-run-i', required=True, type=int, help='subIndx value in mat-file for eeg par file')

# Output info
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--out-dir', required=True, help=f'out file directory')
parser.add_argument('--out-name', required=True, help=f'Name for model')

# HDR Parameters
parser.add_argument('--eeg-sample-frequency', type=int, default=20)
parser.add_argument('--tr', type=int, default=800)
parser.add_argument('--num-trs-skipped-at-beginning', type=int, default=1)
parser.add_argument('--search-type', nargs='+', default=['classic_hemodynamic'], choices=SEARCH_TYPES.keys())
parser.add_argument('--hdr-window', type=float, default=30)
parser.add_argument('--delta-start', type=float, default=1)
parser.add_argument('--delta-end', type=float, default=3, help='inclusive')
parser.add_argument('--delta-step', type=float, default=0.05)
parser.add_argument('--tau-start', type=float, default=0.75)
parser.add_argument('--tau-end', type=float, default=1.75, help='inclusive')
parser.add_argument('--tau-step', type=float, default=0.05)
parser.add_argument('--alpha-start', type=float, default=1.75)
parser.add_argument('--alpha-end', type=float, default=2.25, help='inclusive')
parser.add_argument('--alpha-step', type=float, default=0.05)


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

    models = {}
    for search_type in args.search_type:
        models[f'{search_type}_{args.out_name}'] = SEARCH_TYPES[search_type]['model'](
            eeg,
            fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
            args.out_name,
            args.num_trs_skipped_at_beginning,
            args.hdr_window,
            display_plot=False
        )
        model_for_plotting = SEARCH_TYPES[search_type]['model'](
            eeg,
            fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
            args.out_name,
            args.num_trs_skipped_at_beginning,
            args.hdr_window,
            display_plot=False,
            save_plot_dir=args.out_dir
        )
        save_plot(
            os.path.join(args.out_dir, f'{args.out_name}_across_search_space'),
            plot_eeg_hdr_across_delta_tau_alpha_range,
            eeg,
            args.hdr_window,
            args.tr,
            delta_range,
            tau_range,
            alpha_range,
        )
        model_for_plotting.set_plot_voxels(fmri_voxel_names)
        model_for_plotting.score(PLOT_DELTA, PLOT_TAU, PLOT_ALPHA)

    descriptions, df = search_voxels(models, delta_range, tau_range, alpha_range, args.verbose)

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
