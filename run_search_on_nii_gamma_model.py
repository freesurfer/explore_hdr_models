"""
Run grid search to find best fitting delta, tau, and alpha using the specified hemodynamic model.

By default, uses canonical hemodynamic model:
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0

Usage:
python run_search.py \
    --par_file /autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par \
    --nii-file /autofs/space/ursa_004/users/HDRmodeling/HDRshape/s06_137/rest/fsrest_lh_native/res/res-001.nii.gz \
    --out-file

"""
import argparse
import numpy as np
import os
import pandas as pd

from feeg_fmri_sync import SEARCH_TYPES
from feeg_fmri_sync.io import load_from_nii
from feeg_fmri_sync.constants import EEGData, fMRIData
from feeg_fmri_sync.search import search_voxels

parser = argparse.ArgumentParser()

# EEG file info
parser.add_argument('--par-file', required=True, help=f'par file path')

# fMRI input info
parser.add_argument('--nii-file', required=True, help=f'nii file path')
parser.add_argument('--job-number', default=None, type=int, help='Job number assigned by submit')
parser.add_argument('--number-of-tasks', default=None, type=int, help='Total number of tasks submitted')

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
    fmri_voxel_data, fmri_voxel_names = load_from_nii(args.nii_file, args.job_number, args.number_of_tasks)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    models = {}
    for search_type in args.search_type:
        models[search_type] = SEARCH_TYPES[search_type](
            eeg,
            fMRIData(fmri_voxel_data, args.tr, fmri_voxel_names),
            args.out_name,
            args.num_trs_skipped_at_beginning,
            args.hdr_window,
            plot=False
        )
    delta_range = np.arange(args.delta_start, args.delta_end + args.delta_step, step=args.delta_step)
    tau_range = np.arange(args.tau_start, args.tau_end + args.tau_step, step=args.tau_step)
    alpha_range = np.arange(args.alpha_start, args.alpha_end + args.alpha_step, step=args.alpha_step)

    df, descriptions = search_voxels(models, delta_range, tau_range, alpha_range, args.verbose)

    with open(args.out_file, 'w') as f:
        pd.DataFrame(descriptions).to_csv(f)