"""
Run grid search to find best fitting delta, tau, and alpha using the specified hemodynamic model. Saves results to
mat files and csvs.

Generates QC plots if --max-voxels-safe-for-computation is higher than the number of voxels analyzed

By default, uses canonical hemodynamic model:
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0

Usage for ROI analysis:
python run_search_gamma_model.py \
    --par-file /autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par \
    -v \
    --out-dir /autofs/space/ursa_004/users/HDRmodeling/HDRmodeling/DAN/s06_137-r1/ \
    --out-name DAN_s06_137_r1 \
    roi \
    --mat-file /autofs/space/ursa_004/users/HDRmodeling/ROItcs_fMRI/mat4HMMregWindu.mat \
    --sub-and-run-i 1

Usage for voxel analysis:
python run_search_gamma_model.py \
    --par-file /autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par \
    -v \
    --out-dir /autofs/space/ursa_004/users/HDRmodeling/HDRmodeling/DAN/s06_137-r1/ \
    --out-name DAN_s06_137_r1 \
    nii \
    --nii-file /autofs/space/ursa_004/users/HDRmodeling/HDRshape/s06_137/rest/fsrest_lh_native/res/res-001.nii.gz
"""
import argparse
import numpy as np
import os
import scipy

from feeg_fmri_sync import SEARCH_TYPES
from feeg_fmri_sync.constants import PLOT_ALPHA, PLOT_DELTA, PLOT_TAU, HEMODYNAMIC_MODEL_KEYS
from feeg_fmri_sync.io import load_roi_from_mat, load_from_nii
from feeg_fmri_sync.constants import EEGData, fMRIData
from feeg_fmri_sync.plotting import (
    plot_eeg_hdr_across_delta_tau_alpha_range,
    save_plot
)
from feeg_fmri_sync.search import search_voxels_in_depth_without_df

parser = argparse.ArgumentParser()

########################################################################################################################
#                                                Required arguments
########################################################################################################################
parser.add_argument('--par-file', required=True, help=f'EEG spike train (par file) path')
parser.add_argument('--out-dir', required=True, help=f'out file directory')
# TODO: I suspect there are several potential bugs when --out-name is not a pythonic underscored name. Adding testing
#       would be a good idea
parser.add_argument('--out-name', required=True, help=f'Name for model')

########################################################################################################################
#                                                Optional arguments
########################################################################################################################

# Computational power configuration
parser.add_argument('--max-voxels-safe-for-expensive-computation', type=int, default=40,
                    help='Calculating the significance and plotting each voxel fMRI timecourse is very computationally '
                         'expensive. Automatically raise an error if --get-significance is passed with an fMRI input. '
                         'Turns off plotting if the number of voxels is above this number')

# EEG info
parser.add_argument('--eeg-sample-frequency', type=int, default=20)

# fMRI file info
parser.add_argument('--tr', type=int, default=800)
parser.add_argument('--num-trs-skipped-at-beginning', type=int, default=1)

# Output info
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--get-significance', action='store_true',
                    help='Calculate Pearson\'s Correlation Coefficient and corresponding p-value. '
                         'WARNING: Calculating the significance is very computationally expensive. '
                         'An error will be raised if '
                         'this flag is passed and the fMRI input data has more voxels than '
                         '--max-voxels-safe-for-expensive-computation')

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
parser.add_argument('--standardize-est-fmri', action='store_true')
parser.add_argument('--standardize-input-fmri', action='store_true')
parser.add_argument('--hemodynamic-response-window', type=float, default=30)
# savgol filter
parser.add_argument('--savgol-filter-window-length', type=int, default=5)
parser.add_argument('--savgol-filter-polyorder', type=int, default=5)
parser.add_argument('--deriv', type=int)
parser.add_argument('--delta', type=float)
# gaussian filter
parser.add_argument('--gaussian-filter-sigma', type=float, default=5)
parser.add_argument('--order', type=int)
parser.add_argument('--truncate', type=float)
parser.add_argument('--radius', type=int)
# shared by savgol and gaussian filters
parser.add_argument('--mode', type=str)
parser.add_argument('--cval', type=float)

########################################################################################################################
#                                       Subparsers (for ROI or voxel analysis)
########################################################################################################################

subparsers = parser.add_subparsers(required=True)
mat_file_parser = subparsers.add_parser('roi')
mat_file_parser.add_argument('--mat-file', required=True,
                             help=f'ROI mat file path (must define X and subIndx as variables)', default=None)
mat_file_parser.add_argument('--sub-and-run-i', required=True, type=int,
                             help='subIndx value in mat-file for eeg par file', default=None)

nii_file_parser = subparsers.add_parser('nii')
nii_file_parser.add_argument('--nii-file', required=True, help=f'nii file path', default=None)
# If the runtime on an entire nii file is too large, load_from_nii provides a way of chopping up a large
#   nii file into subsets. These arguments should be uncommented and search_script_writer will need to be updated
#nii_file_parser.add_argument('--job-number', default=None, type=int, help='Job number assigned by submit_from_config')
#nii_file_parser.add_argument('--number-of-tasks', default=None, type=int, help='Total number of tasks submitted by submit_from_config')


if __name__ == '__main__':
    args = parser.parse_args()

    # Load eeg data
    eeg_data = np.fromfile(args.par_file, sep='\n')
    eeg = EEGData(eeg_data, args.eeg_sample_frequency)

    # Load fmri data
    if getattr(args, 'mat_file', None):
        fmri_voxel_data, fmri_voxel_names = load_roi_from_mat(args.mat_file, args.sub_and_run_i)
    elif getattr(args, 'nii_file', None):
        fmri_voxel_data, fmri_voxel_names = load_from_nii(args.nii_file)
        # If the runtime on an entire nii file is too large, load_from_nii provides a way of chopping up a large
        #   nii file into subsets. This command can replace the one from above
        #fmri_voxel_data, fmri_voxel_names = load_from_nii(args.nii_file, args.job_number, args.number_of_tasks)
    else:
        raise RuntimeError("Neither mat-file nor nii-file was provided. This code should be unreachable!")

    if len(fmri_voxel_data.shape) == 1:
        safe_to_be_computationally_expensive = 1 < args.max_voxels_safe_for_expensive_computation
    else:
        safe_to_be_computationally_expensive = fmri_voxel_data.shape[0] < args.max_voxels_safe_for_expensive_computation

    if args.get_significance and not safe_to_be_computationally_expensive:
        raise RuntimeError(f'Cannot calculate significance for {fmri_voxel_data.shape[0]} voxels. '
                           f'--max-voxels-safe-for-expensive-computation is set to '
                           f'{args.max_voxels_safe_for_expensive_computation}. '
                           f'Either increase --max-voxels-safe-for-expensive-computation or remove --get-significance')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    delta_range = np.arange(args.delta_start, args.delta_end + args.delta_step, step=args.delta_step)
    tau_range = np.arange(args.tau_start, args.tau_end + args.tau_step, step=args.tau_step)
    alpha_range = np.arange(args.alpha_start, args.alpha_end + args.alpha_step, step=args.alpha_step)

    search_kwargs = {kn: kv for kn, kv in vars(args).items() if kn in HEMODYNAMIC_MODEL_KEYS}
    models = {f'{args.search_type}_{args.out_name}': SEARCH_TYPES[args.search_type]['model'](
        eeg,
        fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
        args.out_name,
        args.num_trs_skipped_at_beginning,
        display_plot=False,
        **search_kwargs
    )}
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
    if safe_to_be_computationally_expensive:
        # Create a model to plot actual fMRI vs estimated
        model_for_plotting = SEARCH_TYPES[args.search_type]['model'](
            eeg,
            fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
            args.out_name,
            args.num_trs_skipped_at_beginning,
            display_plot=False,
            save_plot_dir=args.out_dir,
            **search_kwargs
        )
        model_for_plotting.set_plot_voxels(fmri_voxel_names)
        model_for_plotting.score(PLOT_DELTA, PLOT_TAU, PLOT_ALPHA)

    data, variable_names, indexers = search_voxels_in_depth_without_df(models, delta_range, tau_range, alpha_range,
                                                                       args.verbose, args.get_significance)

    for v_i, variable_name in enumerate(variable_names):
        out_name = f'{variable_name}_search_on_' \
                   f'{os.path.basename(args.mat_file).split(".")[0]}_sub{args.sub_and_run_i}_{args.out_name}.mat'
        if args.verbose:
            print(f'Writing {variable_name} search to {out_name}')
        scipy.io.savemat(
            os.path.join(args.out_dir, out_name),
            {variable_name: data.take(v_i, axis=(len(data.shape) - 1))}
        )

    for i, (indexer_name, indexer_values) in enumerate(indexers):
        out_name = f'{i}_key_{indexer_name}_for_{os.path.basename(args.mat_file).split(".")[0]}' \
                   f'_sub{args.sub_and_run_i}_{args.out_name}.csv'
        np.savetxt(os.path.join(args.out_dir, out_name), indexer_values, delimiter=",", fmt="%s")
