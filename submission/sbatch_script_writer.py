from typing import List, Generator, Dict, Tuple

from submission.script_writers import ScriptWriter
from submission.search_script_writer import SearchScriptWriter
from submission.parse_config import get_config_section, get_items_for_section_ignoring_defaults


class SBatchWriter(ScriptWriter):
    def __init__(self, config):
        self.sbatch_options: List[Tuple[str, str]] = get_items_for_section_ignoring_defaults(config, 'sbatch')

    def get_lines(self):
        lines = []
        for key, value in self.sbatch_options:
            lines.append(f'#SBATCH --{key}={value}')
        return lines


class WriteSubmissionSh:
    def __init__(self, sbatch: SBatchWriter, script_setup: SearchScriptWriter, conda_env='feeg_fmri'):
        self.sbatch = sbatch
        self.conda_env = conda_env
        self.script_setup = script_setup

    @staticmethod
    def get_conda_lines() -> List[str]:
        return [
            '',
            '# >>> conda initialize >>>',
            '__conda_setup="$(\'/usr/pubsw/packages/python/anaconda3-2019.03/bin/conda\' \'shell.bash\' \'hook\' 2> '
            '/dev/null)"',
            'if [ $? -eq 0 ]; then',
            '\teval "$__conda_setup"',
            'else',
            '\tif [ -f "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh" ]; then',
            '\t\t. "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh"'
            '\telse',
            '\t\texport PATH="/usr/pubsw/packages/python/anaconda3-2019.03/bin:$PATH"',
            '\tfi',
            'fi',
            'unset __conda_setup',
            '# <<< conda initialize <<<',
            ''
        ]

    def get_identifiers(self) -> Generator[int, None, None]:
        yield self.script_setup.get_identifiers()

    def write_file(self, identifier: int, out_name: str) -> str:
        lines = ['#!/bin/bash']
        lines.extend(self.sbatch.get_lines())
        lines.extend(self.get_conda_lines())
        lines.append(f'conda activate {self.conda_env}')
        lines.extend(self.script_setup.get_lines_for_identifier(identifier))
        lines = [f'{line}\n' for line in lines]
        with open(out_name, 'w') as f:
            f.writelines(lines)
        return out_name
