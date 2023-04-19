import inspect

from feeg_fmri_sync.simulations import (
    generate_downsampled_simulated_fmri,
    generate_summed_simulated_fmri
)
from feeg_fmri_sync.models import (
    CanonicalHemodynamicModel,
    VectorizedSumEEGHemodynamicModel
)


SEARCH_TYPES = {
    'classic_hemodynamic': {
        'model': CanonicalHemodynamicModel,
        'simulation_generator': generate_downsampled_simulated_fmri,
    },
    'classic_hemodynamic_sum': {
        'model': VectorizedSumEEGHemodynamicModel,
        'simulation_generator': generate_summed_simulated_fmri,
    },
}

VALID_KWARGS = [
    inspect.signature(model['model']) for model in SEARCH_TYPES.values()
]
