"""
submit sbatch command
"""

import argparse
import glob
import os
import re
import subprocess

from typing import Optional, List, Dict, Any

from feeg_fmri_sync import SEARCH_TYPES, HemodynamicModel
from feeg_fmri_sync.constants import (
    EEG_DIR,
    FMRI_DIR,
    MLSC_ROOT_DIR,
    PROJECT_DIR,
    URSA_ROOT_DIR
)
from feeg_fmri_sync.simulations import ModelsToTest
from feeg_fmri_sync.utils import get_fmri_filepaths


def build_models_to_test(search_types: List[str],
                         search_names: Optional[List[str]] = None) -> List[ModelsToTest]:
    models_to_test = []
    if not search_names:
        search_names = search_types
    if len(search_names) != len(search_types):
        raise ValueError(f'search_names ({search_names}) must be the same length as search_types: ({search_types})')
    for search_type, search_name in zip(search_types, search_names):
        models_to_test.append({
            'name': search_name,
            'model': SEARCH_TYPES[search_type]['model'],
            'fmri_data_generator': SEARCH_TYPES[search_type]['simulation_generator']
        })
    return models_to_test


parser = argparse.ArgumentParser(
    description='Wrapper command to submit a sbatch job . **Assumes EEG and fMRI data exist for the same subjects**'
)

parser.add_argument('--network', action='append',
                    help=f'EEG network to analyze. By default, analyze all networks listed in {PROJECT_DIR}/{EEG_DIR}')
parser.add_argument('-s', '--subject', action='append',
                    help=f'Subject to analyze. By default, analyze all subjects listed in {PROJECT_DIR}{FMRI_DIR}')
parser.add_argument('-r', '--run', action='append',
                    help=f'Run to analyze. By default, analyze all runs for each subject')
parser.add_argument('--hemisphere', choices=['l', 'r'], action='append',
                    help=f'fMRI hemisphere to analyze. By default, analyze both hemispheres for each subject')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--mlsc', action='store_true', help='Set to true if this is being run on MLSC')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--out', required=True, help=f'Directory to write results to. Will be in {PROJECT_DIR}/. ' +
                                                 'Resulting filename will be <subject>-<network>-<hemisphere>h-r<run>-j<job-number>.csv (start and end are defined by the job number)')
parser.add_argument('--search-type', default='classic_hemodynamic', choices=SEARCH_TYPES.keys())

parser.add_argument('--account', default='mobius')
parser.add_argument('--cpus-per-task', default=1)
parser.add_argument('--time', default='0:15:00')

args = parser.parse_args()

root_dir = MLSC_ROOT_DIR if args.mlsc else URSA_ROOT_DIR

# By default, get all networks
if not args.network:
    args.network = []
    for d in os.scandir(os.path.join(root_dir, PROJECT_DIR, EEG_DIR)):
        if d.is_dir():
            args.network.append(d.name)

# By default, get all subjects
if not args.subject:
    args.subject = []
    for d in os.scandir(os.path.join(root_dir, PROJECT_DIR, FMRI_DIR)):
        if d.is_dir():
            args.subject.append(d.name)

for network in args.network:
    for subject in args.subject:
        # By default get all runs
        eeg_filenames = []
        if not args.run:
            eeg_filenames.append(f'{subject}-r*.par')
        else:
            for run in args.run:
                eeg_filenames.append(f'{subject}-r{run}.par')

        eeg_files = []
        for eeg_filename in eeg_filenames:
            eeg_files.extend(glob.glob(os.path.join(root_dir, PROJECT_DIR, EEG_DIR, network, eeg_filename)))

        for eeg_file in eeg_files:
            run = re.search(f'{subject}-r([0-9]+).par', eeg_file)
            if not run:
                print(f'Unable to find run number in filename: {eeg_file}. Skipping...')
                continue
            # By default get all hemisphere
            fmri_files = get_fmri_filepaths(root_dir, subject, args.hemisphere, run)

### Inputs
hemisphere = 'l'
network = 'DAN5'
run = '1'
subject = 's06_137'
in_mlsc = True
write_dir = 'HDRmodeling'
job_number = 0
verbose = True

### Code starts

par_f = os.path.join(root_dir, PROJECT_DIR, EEG_DIR, network, f'{subject}-r{run}.par')
nii_f = os.path.join(root_dir, PROJECT_DIR, FMRI_DIR, subject, 'rest', f'fsrest_{hemisphere}h_native', 'res',
                     f'res-00{run}.nii.gz')

subprocess.check_call(f"""
#!/bin/bash
#SBATCH --account={args.account}
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem=64G
#SBATCH --time={args.time}
#SBATCH --output=slurm-%A_%a.out

time python ./scratch.py
""")
