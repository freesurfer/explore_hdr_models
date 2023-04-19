import inspect

from feeg_fmri_sync.models import (
    HemodynamicModel,
    SumEEGHemodynamicModel
) 
from feeg_fmri_sync.simulations import (
    generate_downsampled_simulated_fmri,
    generate_summed_simulated_fmri
)
from feeg_fmri_sync.vectorized_models import (
    VectorizedHemodynamicModel,
    VectorizedSumEEGHemodynamicModel
)


SEARCH_TYPES = {
    'classic_hemodynamic': {
        'model': VectorizedHemodynamicModel,
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
