from configparser import ConfigParser, SectionProxy
from typing import Optional, List


def get_config_subsection_variable(config: ConfigParser, varname: str, section: Optional[str] = None) -> str:
    if not section:
        section_with_variables = config.defaults()
        if varname not in section_with_variables:
            raise ValueError(f'Unable to locate {varname} variable in DEFAULT section')
        return section_with_variables[varname]
    return config.get(section, varname)


def get_config_section(config: ConfigParser, section: str) -> SectionProxy:
    if section not in config:
        raise ValueError(f'Unable to locate section "{section}"')
    return config[section]


def get_values_for_section_ignoring_defaults(config: ConfigParser, section: str) -> List[str]:
    retlist = []
    defaults = config.defaults()
    for key, value in get_config_section(config, section).items():
        if key in defaults:
            continue
        retlist.append(value)
    return retlist


def get_root(config: ConfigParser, location: Optional[str]) -> str:
    if location:
        return get_config_subsection_variable(config, 'root-dir', f'location.{location}')
    return get_config_subsection_variable(config, 'root-dir')

