import itertools
import os
import pathlib
import re
from abc import ABCMeta
from configparser import ConfigParser
from typing import List, Dict, Generator, Optional, Type, TypeVar, Any

from feeg_fmri_sync.constants import HEMODYNAMIC_MODEL_KEYS, SEARCH_KEYS
from submission.parse_config import (
    get_config_section,
    get_config_subsection_variable,
    get_items_for_section_ignoring_defaults, get_items_for_section_given_keys, get_config_subsection_boolean
)
from submission.script_writers import ScriptWriter, IterativeScriptWriter
from feeg_fmri_sync.utils import get_fmri_filepaths, get_i_for_subj_and_run


class HDRSearch(IterativeScriptWriter, metaclass=ABCMeta):
    type_str: str
    lookup_str: str
    out_name: str

    @staticmethod
    def get_out_name(identifier) -> str:
        return identifier


HDRSearch_subclass = TypeVar('HDRSearch_subclass', bound=HDRSearch)


class FMRIFiles(IterativeScriptWriter, metaclass=ABCMeta):
    type_str: str
    script_path: str
    variables: Dict[str, str]
    out_name: str

    def get_out_name(self, identifier) -> str:
        id_str = self.get_str_for_identifier(identifier)
        return f'{self.out_name}_{id_str}' if id_str else self.out_name

    def get_str_for_identifier(self, identifier) -> str:
        return ''


fMRIFiles_subclass = TypeVar('fMRIFiles_subclass', bound=FMRIFiles)


class SearchScriptWriter(IterativeScriptWriter):
    def __init__(self,
                 fmri_files: fMRIFiles_subclass,
                 hdr_search: HDRSearch_subclass,
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

    def get_lines_for_identifier(self, identifier: Any) -> List[str]:
        fmri_file_identifier, hdr_analysis_identifier = identifier
        lines = [f'time python {self.fmri_files.script_path} \\',
                 f'\t--par-file={self.par_file} \\',
                 f'\t--out-dir={self.out_dir}/{self.fmri_files.get_str_for_identifier(fmri_file_identifier)} \\']
        lines.extend([f'\t{line} \\' for line in self.hdr_analysis.get_lines_for_identifier(hdr_analysis_identifier)])
        if self.verbose:
            lines.append('\t--verbose \\')
        if self.par_network:
            out_name = f'{self.fmri_files.get_out_name(fmri_file_identifier)}_{self.par_network}'
        else:
            out_name = self.fmri_files.get_out_name(fmri_file_identifier)
        lines.append(f'\t--out-name={out_name}_{self.hdr_analysis.get_out_name(hdr_analysis_identifier)}')
        lines.extend([f'\t{line} \\' for line in self.fmri_files.get_lines_for_identifier(fmri_file_identifier)])
        return lines

    def get_identifiers(self) -> Generator[Any, None, None]:
        for identifiers in itertools.product(self.fmri_files.get_identifiers(), self.hdr_analysis.get_identifiers()):
            yield identifiers

    @staticmethod
    def get_str_for_identifier(identifier) -> str:
        return identifier[1]


class GammaCanonicalHDR(HDRSearch):
    type_str = 'gamma-canonical-hdr'
    lookup_str = f'modeling-type.{type_str}'
    search_section_keys = SEARCH_KEYS
    model_section_keys = ['-'.join(s.split('_')) for s in HEMODYNAMIC_MODEL_KEYS] + ['search-type']
    boolean_flags = ['standardize-est-fmri', 'standardize-input-fmri', 'save-data-to-mat', 'get-significance']

    def __init__(self, config):
        search_variables = get_items_for_section_given_keys(config, self.lookup_str, self.search_section_keys)
        self.search_variables = {}
        search_types = {}
        for varname, variable in search_variables:
            if varname == 'search-types':
                for search_type in variable.strip().split(','):
                    search_type = search_type.strip()
                    model_kwargs = get_items_for_section_given_keys(
                        config, f'{self.lookup_str}.{search_type}', self.model_section_keys
                    )
                    model_kwargs_to_save = []
                    for model_varname, model_variable in model_kwargs:
                        if model_varname in self.boolean_flags:
                            model_kwargs_to_save.append(
                                (model_varname,
                                 get_config_subsection_boolean(config, model_varname, f'{self.lookup_str}.{search_type}'))
                            )
                        else:
                            model_kwargs_to_save.append((model_varname, model_variable))
                    search_types[search_type] = model_kwargs_to_save
            else:
                if varname in self.boolean_flags:
                    self.search_variables[varname] = get_config_subsection_boolean(config, varname, self.lookup_str)
                else:
                    self.search_variables[varname] = variable
        self.search_types = search_types

    def get_varname_variable(self, varname, variable):
        if varname in self.boolean_flags:
            if variable:
                return f'--{varname}'
            return None
        return f'--{varname}={variable}'

    def get_lines_for_identifier(self, identifier: Any) -> List[str]:
        specific_search_variables = self.search_types[identifier]
        lines = []
        for varname, variable in self.search_variables.items():
            new_line = self.get_varname_variable(varname, variable)
            if new_line:
                lines.append(new_line)
        for varname, variable in specific_search_variables:
            new_line = self.get_varname_variable(varname, variable)
            if new_line:
                lines.append(new_line)
        return lines

    def get_identifiers(self) -> Generator[Any, None, None]:
        for key in self.search_types.keys():
            yield key


class NiiFMRIFiles(FMRIFiles):
    type_str = 'nii'
    script_path = str(pathlib.Path(__file__).parent.with_name('run_search_gamma_model.py').resolve())
    # TODO: decide about job-number and number-of-tasks

    def __init__(self, config: ConfigParser, subject: str, run: int, root_dir: str, **kwargs):
        fmri_dir = get_config_subsection_variable(config, 'fmri-dir', section=f"fmri-input.{self.type_str}")
        # By default, get all hemispheres
        hemispheres = [get_config_section(config, 'fmri-input.nii').get('hemisphere')]
        self.fmri_filenames = get_fmri_filepaths(os.path.join(root_dir, fmri_dir), subject, hemispheres, run)
        self.out_name = f'{subject}_r{run}'

    def get_identifiers(self) -> Generator[int, None, None]:
        for i in range(len(self.fmri_filenames)):
            yield i

    def get_lines_for_identifier(self, fmri_i):
        return ['nii', f'--nii-file={self.fmri_filenames[fmri_i]}']

    def get_str_for_identifier(self, identifier) -> str:
        return re.search(r'fsrest_([lr]h)_native', self.fmri_filenames[identifier]).group(1)


class RoiFile(FMRIFiles):
    type_str = 'roi'
    script_path = str(pathlib.Path(__file__).parent.with_name('run_search_gamma_model.py').resolve())

    def __init__(self, config: ConfigParser, subject: str, run: int, root_dir: str,
                 all_subjects_list: List[str], **kwargs):
        self.mat_file = os.path.join(
            root_dir,
            get_config_subsection_variable(config, 'mat-file', section=f'fmri-input.{self.type_str}')
        )
        self.subj_and_run_i = get_i_for_subj_and_run(subject, str(run), all_subjects_list) + 1
        self.out_name = f'{subject}_r{run}'

    def get_identifiers(self) -> Generator[int, None, None]:
        yield 0

    def get_lines_for_identifier(self, ind):
        return [
            f'roi',
            f'--mat-file={self.mat_file}',
            f'--sub-and-run-i={self.subj_and_run_i}'
        ]


TYPE_STR_TO_FMRIFILE_CLASS = {
    klass.type_str: klass for klass in [NiiFMRIFiles, RoiFile]
}


def get_fmri_files_creator(config: ConfigParser) -> Type[fMRIFiles_subclass]:
    return TYPE_STR_TO_FMRIFILE_CLASS[get_config_subsection_variable(config, 'fmri-input')]


TYPE_STR_TO_HDRSEARCH_CLASS = {
    klass.type_str: klass for klass in [GammaCanonicalHDR]
}


def get_hdr_search_creator(config: ConfigParser) -> Type[HDRSearch_subclass]:
    return TYPE_STR_TO_HDRSEARCH_CLASS[get_config_subsection_variable(config, 'modeling-type')]
