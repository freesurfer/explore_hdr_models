import os
import pathlib
import re
from abc import ABC, abstractmethod
from configparser import ConfigParser
from typing import List, Dict, Generator, Optional, Type

from feeg_fmri_sync.submission.parse_config import get_config_section, get_config_subsection_variable
from feeg_fmri_sync.submission.script_writers import ScriptWriter, IterativeScriptWriter
from feeg_fmri_sync.utils import get_fmri_filepaths, get_i_for_subj_and_run


class HDRSearch(ABC, ScriptWriter):
    lookup_str: str


class FMRIFiles(ABC, IterativeScriptWriter):
    type_str: str
    script_path: str
    variables: Dict[str, str]
    out_name: str

    @abstractmethod
    def get_out_name(self, identifier) -> str:
        id_str = self.get_str_for_identifier(identifier)
        return f'{self.out_name}_{id_str}' if id_str else self.out_name

    @abstractmethod
    def get_str_for_identifier(self, identifier) -> str:
        return ''


class SearchScriptWriter(IterativeScriptWriter):
    def __init__(self,
                 fmri_files: FMRIFiles,
                 hdr_search: HDRSearch,
                 config: ConfigParser,
                 root_dir: str,
                 par_file: str,
                 par_network: Optional[str] = None,
                 out_dir: Optional[str] = None):
        self.fmri_files = fmri_files
        self.hdr_analysis = hdr_search
        self.par_file = par_file
        self.out_dir = out_dir
        if not out_dir:
            self.out_dir = os.path.join(
                root_dir,
                get_config_subsection_variable(config, 'out-dir')
            )
        self.verbose = config.getboolean(self.hdr_analysis.lookup_str, 'verbose', fallback=False)
        self.par_network = par_network

    def get_lines_for_identifier(self, identifier) -> List[str]:
        lines = [f'time python {self.fmri_files.script_path} \\'
                 f'--par-file={self.par_file} \\',
                 f'--out-dir={self.out_dir}/{self.fmri_files.get_str_for_identifier(identifier)} \\']
        lines.extend([f'{line} \\' for line in self.fmri_files.get_lines_for_identifier(identifier)])
        lines.extend([f'{line} \\' for line in self.hdr_analysis.get_lines()])
        if self.verbose:
            lines.append('--verbose \\')
        if self.par_network:
            out_name = f'{self.fmri_files.get_out_name(identifier)}_{self.par_network}'
        else:
            out_name = self.fmri_files.get_out_name(identifier)
        lines.append(f'--out-name={out_name}')
        return lines

    def get_identifiers(self) -> Generator[int]:
        yield self.fmri_files.get_identifiers()


class GammaCanonicalHDR(HDRSearch):
    lookup_str = f'modeling-type.gamma-canonical-hdr'

    def __init__(self, config):
        self.search_variables = get_config_section(config, self.lookup_str)
        search_types = get_config_section(config, f'{self.lookup_str}.search-type')
        self.search_types = [search_type for search_type in search_types.items()]

    def get_lines(self) -> List[str]:
        lines = [f'--search-type={search_type}' for search_type in self.search_types]
        for varname, variable in self.search_variables.items():
            lines.append(f'--{varname}={variable}')
        return lines


class NiiFMRIFiles(FMRIFiles):
    type_str = 'nii'
    script_path = str(pathlib.Path(__file__).with_name('run_search_on_nii_gamma_model.py').resolve())
    # TODO: decide about job-number and number-of-tasks

    def __init__(self, config: ConfigParser, subject: str, run: int, root_dir: str, **kwargs):
        fmri_dir = get_config_subsection_variable(config, 'fmri-dir', section=f"fmri-input.{self.type_str}")
        # By default, get all hemispheres
        hemispheres = [get_config_section(config, 'fmri-input.nii').get('hemisphere')]
        self.fmri_filenames = get_fmri_filepaths(os.path.join(root_dir, fmri_dir), subject, hemispheres, run)
        self.out_name = f'{subject}_r{run}'

    def get_identifiers(self) -> Generator[int]:
        for i in range(len(self.fmri_filenames)):
            yield i

    def get_lines(self, fmri_i):
        return [f'--nii-file={self.fmri_filenames[fmri_i]}']

    def get_str_for_identifier(self, identifier) -> str:
        return re.search(r'fsrest_([lr]h)_native', self.fmri_filenames[identifier]).group(1)


class RoiFile(FMRIFiles):
    type_str = 'roi'
    script_path = str(pathlib.Path(__file__).with_name('run_search_on_roi_gamma_model.py').resolve())

    def __init__(self, config: ConfigParser, subject: str, run: int, root_dir: str,
                 all_subjects_list: List[str], **kwargs):
        self.mat_file = os.path.join(
            root_dir,
            get_config_subsection_variable(config, 'mat-file', section=f'fmri-input.{self.type_str}')
        )
        self.subj_and_run_i = get_i_for_subj_and_run(subject, str(run), all_subjects_list)
        self.out_name = f'{subject}_r{run}'

    def get_identifiers(self) -> Generator[int]:
        yield 0

    def get_lines(self, ind):
        return [
            f'--mat-file={self.mat_file}',
            f'--sub-and-run-i={self.subj_and_run_i}'
        ]


TYPE_STR_TO_FMRIFILE_CLASS = {
    klass.type_str: klass for klass in [NiiFMRIFiles, RoiFile]
}


def get_fmri_files_creator(config: ConfigParser) -> Type[FMRIFiles]:
    return TYPE_STR_TO_FMRIFILE_CLASS[get_config_subsection_variable(config, 'fmri-input')]
