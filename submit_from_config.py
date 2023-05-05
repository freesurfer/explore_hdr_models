"""
submit sbatch command
"""
import argparse
import configparser
import glob
import os
import re
import subprocess
from typing import Dict, Tuple

from submission.parse_config import (
    get_root,
    get_config_subsection_variable,
    get_values_for_section_ignoring_defaults,
)
from submission.search_script_writer import (
    get_fmri_files_creator,
    SearchScriptWriter,
    get_hdr_search_creator
)
from submission.sbatch_script_writer import WriteSubmissionSh
from submission.submission_params_writer import get_submission_writer_creator

parser = argparse.ArgumentParser(
    description='Wrapper command to run a search from a config file. '
                '**ASSUMES EEG and fMRI data EXIST for the SAME SUBJECTS**'
)
parser.add_argument('config', help=f'Configuration file to read parameters')
parser.add_argument('--location', default=None, help=f'Location section to pull root directory from')
parser.add_argument('--only-files', action='store_true', help='Only generate files, do not submit them to the cluster')


if __name__ == '__main__':
    args = parser.parse_args()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config)
    # Set root dir
    root_dir = get_root(config, args.location)

    out_dir = get_config_subsection_variable(config, 'out-dir')

    # Determine how we will run the script
    how = get_config_subsection_variable(config, 'how')
    script_out_dir = os.path.join(root_dir, out_dir, 'submission_scripts')
    if not os.path.exists(script_out_dir):
        os.makedirs(script_out_dir)
    job_name_to_out_file_paths: Dict[str, Tuple[WriteSubmissionSh, str]] = {}

    # Grab inputs specific for queueing engine
    submission_header_writer = get_submission_writer_creator(config)(config)
    # Grab search parameters
    hdr_search = get_hdr_search_creator(config)(config)
    # Get type of fmri files we expect
    fmri_files_klass = get_fmri_files_creator(config)
    # Get conda environment search command should be run in
    conda_env = config.defaults().get('conda_env', 'feeg_fmri')

    # Get directory with networks for EEG par files
    network_dir = get_config_subsection_variable(config, 'network-dir')
    # Get list of networks if specified
    networks = [network_name for network_name in get_values_for_section_ignoring_defaults(config, 'networks')]
    # By default, get all networks
    if not networks:
        networks = []
        for d in os.scandir(os.path.join(root_dir, network_dir)):
            if d.is_dir():
                networks.append(d.name)

    for network in networks:
        # Get list of subjects if specified
        subjects = [subject_name for subject_name in get_values_for_section_ignoring_defaults(config, 'subjects')]
        # Get list of runs if specified
        runs = [run_name for run_name in get_values_for_section_ignoring_defaults(config, 'runs')]
        # Get list of all subjects (in case we're doing roi analysis)
        all_subjects_and_runs_list = []
        subject_names_list = []
        for d in os.scandir(os.path.join(root_dir, network_dir, network)):
            try:
                subject_names_list.append(re.search('(s[0-9_A-Za-z-]+)-r[0-9]+\.par', d.name).group(1))
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
            if len(eeg_files) == 0:
                print(f'Unable to find matching eeg files for {eeg_filenames} in {os.path.join(root_dir, network_dir, network)}')
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
                submission_sh_file_writer = WriteSubmissionSh(submission_header_writer, search_script_writer, conda_env)
                for identifier in submission_sh_file_writer.get_identifiers():
                    search_type = submission_sh_file_writer.get_identifier_string(identifier)
                    job_name_to_out_file_paths[f'{network}_{subject}_r{run}_search_{search_type}'] = (
                        submission_sh_file_writer,
                        submission_sh_file_writer.write_file(
                            identifier,
                            os.path.join(root_dir, script_out_dir, f'{network}_{subject}_r{run}_search_{search_type}_script.sh')
                        ))
    if not args.only_files:
        processes = []
        for job_name, (file_writer, script_path) in job_name_to_out_file_paths.items():
            processes.append(
                subprocess.run(file_writer.get_subprocess_command(job_name, script_path), capture_output=True, check=True)
            )
        submitted_jobs = []
        for process in processes:
            out = process.stdout.decode("utf8")
            print(out.strip())
            job_id_match = re.search(r'([0-9]+)', out)
            if job_id_match:
                submitted_jobs.append(f'{job_id_match.group(1)}\n')

        with open(os.path.join(root_dir, script_out_dir, 'submitted_jobs.txt'), 'w') as f:
            f.writelines(submitted_jobs)

