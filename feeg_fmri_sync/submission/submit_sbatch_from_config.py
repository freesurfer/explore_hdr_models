"""
submit sbatch command
"""
import argparse
import configparser
import glob
import os
import re
import subprocess
from typing import Dict

from feeg_fmri_sync.submission.parse_config import get_root, get_config_subsection_variable, get_config_section
from feeg_fmri_sync.submission.search_script_writer import HDRSearch, get_fmri_files_creator, SearchScriptWriter
from feeg_fmri_sync.submission.sbatch_script_writer import SBatchWriter, WriteSubmissionSh


parser = argparse.ArgumentParser(
    description='Wrapper command to submit a sbatch job . **Assumes EEG and fMRI data exist for the same subjects**'
)
parser.add_argument('config', help=f'Configuration file to read parameters')
parser.add_argument('--location', default=None, help=f'Location section to pull root directory from')
parser.add_argument('--only-files', action='store_true', help='Only generate files, do not submit them to the cluster')


if __name__ == '__main__':
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    # Set root dir
    root_dir = get_root(config, args.location)

    # Set up place to write sbatch scripts
    out_dir = get_config_subsection_variable(config, 'out-dir')
    sbatch_out_dir = os.path.join(root_dir, out_dir, 'sbatch_scripts')
    if not os.path.exists(sbatch_out_dir):
        os.makedirs(sbatch_out_dir)
    sbatch_job_name_to_out_files: Dict[str, str] = {}

    # Grab SBATCH inputs
    sbatch_writer = SBatchWriter(config)
    # Grab search parameters
    hdr_search = HDRSearch(config)
    # Get type of fmri files we expect
    fmri_files_klass = get_fmri_files_creator(config)
    # Get conda environment search command should be run in
    conda_env = config.defaults().get('conda_env', 'feeg_fmri')

    # Get directory with networks for EEG par files
    network_dir = get_config_subsection_variable(config, 'network-dir', 'DEFAULT')
    # Get list of networks if specified
    networks = [network_name for network_name in get_config_section(config, 'networks').values()]
    # By default, get all networks
    if not networks:
        networks = []
        for d in os.scandir(os.path.join(root_dir, network_dir)):
            if d.is_dir():
                networks.append(d.name)

    for network in networks:
        # Get list of subjects if specified
        subjects = [subject_name for subject_name in get_config_section(config, 'subjects').values()]
        # Get list of runs if specified
        runs = [run_name for run_name in get_config_section(config, 'runs').values()]
        # Get list of all subjects (in case we're doing roi analysis)
        all_subjects_and_runs_list = []
        subject_names_list = []
        for d in os.scandir(os.path.join(root_dir, network_dir, network)):
            try:
                subject_names_list.append(re.search('s[0-9]+_', d.name).group(1))
                all_subjects_and_runs_list.append(d.name)
            except AttributeError:
                pass
        # By default, get all subjects
        all_subjects_and_runs_list.sort()
        if not subjects:
            subjects = subject_names_list
        for subject in subjects:
            eeg_filenames = []
            # By default, get all runs
            if not runs:
                eeg_filenames.append(f'{subject}-r*.par')
            else:
                for run in runs:
                    eeg_filenames.append(f'{subject}-r{run}.par')
            eeg_files = []
            for eeg_filename in eeg_filenames:
                eeg_files.extend(glob.glob(os.path.join(root_dir, network_dir, network, eeg_filename)))
        
            for eeg_file in eeg_files:
                try:
                    run = re.search(f'{subject}-r([0-9]+).par', eeg_file).group(1)
                except AttributeError:
                    print(f'Unable to find run number in filename: {eeg_file}. Skipping...')
                    continue
                # Get all fmri files/data associated with this subject (important for hemisphere parsing on nii files)
                fmri_files = fmri_files_klass(
                    config=config,
                    subject=subject,
                    run=int(run),
                    root_dir=root_dir,
                    all_subjects_list=all_subjects_and_runs_list
                )
                search_script_writer = SearchScriptWriter(
                    fmri_files,
                    hdr_search,
                    config,
                    root_dir,
                    eeg_file,
                    network,
                    out_dir=os.path.join(root_dir, out_dir, network, subject, run)
                )
                sbatch_sh_file_writer = WriteSubmissionSh(sbatch_writer, search_script_writer, conda_env)
                for identifier in sbatch_sh_file_writer.get_identifiers():
                    sbatch_job_name_to_out_files[f'{network}_{subject}_r{run}'] = sbatch_sh_file_writer.write_file(
                        identifier,
                        os.path.join(root_dir, sbatch_out_dir, f'{network}_{subject}_r{run}_sbatch_script.sh')
                    )
    if not args.only_files:
        processes = []
        for job_name, sbatch_script in sbatch_job_name_to_out_files.items():
            processes.append(subprocess.check_call(['sbatch', f'--job-name={job_name}', sbatch_script]))

