import inspect

from feeg_fmri_sync.simulations import (
    generate_downsampled_simulated_fmri,
    generate_summed_simulated_fmri
)
from feeg_fmri_sync.models import (
    CanonicalHemodynamicModel,
    GaussianFilterHemodynamicModel,
    HemodynamicModelSumEEG,
    SavgolFilterHemodynamicModel
)


SEARCH_TYPES = {
    'classic_hemodynamic': {
        'model': CanonicalHemodynamicModel,
        'simulation_generator': generate_downsampled_simulated_fmri,
    },
    'classic_hemodynamic_sum': {
        'model': HemodynamicModelSumEEG,
        'simulation_generator': generate_summed_simulated_fmri,
    },
    'classic_hemodynamic_gaussian_filter': {
        'model': GaussianFilterHemodynamicModel,
        'simulation_generator': generate_downsampled_simulated_fmri,
    },
    'classic_hemodynamic_savgol_filter': {
        'model': SavgolFilterHemodynamicModel,
        'simulation_generator': generate_downsampled_simulated_fmri,
    }
}

VALID_KWARGS = [
    inspect.signature(model['model']) for model in SEARCH_TYPES.values()
]
