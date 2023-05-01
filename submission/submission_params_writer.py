import subprocess
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from typing import List, Tuple, TypeVar, Type

from submission.parse_config import get_items_for_section_ignoring_defaults, get_config_subsection_variable
from submission.script_writers import ScriptWriter


class SubmissionWriter(ScriptWriter, metaclass=ABCMeta):
    type_str: str
    lookup_str: str

    @abstractmethod
    def get_subprocess_command(self, job_name: str, script_path: str) -> List[str]:
        raise NotImplementedError


SubmissionWriter_subclass = TypeVar('SubmissionWriter_subclass', bound=SubmissionWriter)


class SBatchWriter(SubmissionWriter):
    type_str = 'sbatch'
    lookup_str = 'sbatch'

    def __init__(self, config):
        self.sbatch_options: List[Tuple[str, str]] = get_items_for_section_ignoring_defaults(config, self.lookup_str)

    def get_lines(self):
        lines = []
        for key, value in self.sbatch_options:
            lines.append(f'#SBATCH --{key}={value}')
        return lines

    def get_subprocess_command(self, job_name: str, script_path: str) -> List[str]:
        return ['sbatch', f'--job-name={job_name}', script_path]


class PbScriptWriter(SubmissionWriter):
    type_str = 'pbsubmit'
    lookup_str = 'pbsubmit'

    def __init__(self, config):
        self.sbatch_options: List[Tuple[str, str]] = get_items_for_section_ignoring_defaults(config, self.lookup_str)

    def get_lines(self):
        lines = []
        for key, value in self.sbatch_options:
            lines.append(f'#SBATCH --{key}={value}')
        return lines

    def get_subprocess_command(self, job_name: str, script_path: str) -> List[str]:
        subprocess.run(['chmod', '+x', script_path], check=True)
        return ['pbsubmit', '-c', f'"{script_path}"']


TYPE_STR_TO_SUBMISSION_WRITER_CLASS = {
    klass.type_str: klass for klass in [SBatchWriter, PbScriptWriter]
}


def get_submission_writer_creator(config: ConfigParser) -> Type[SubmissionWriter_subclass]:
    return TYPE_STR_TO_SUBMISSION_WRITER_CLASS[get_config_subsection_variable(config, 'how')]
