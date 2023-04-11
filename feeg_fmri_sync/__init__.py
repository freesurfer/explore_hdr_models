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
        'simulation': generate_downsampled_simulated_fmri,
    },
    'hemodynamic_sum_eeg': {
        'model': VectorizedSumEEGHemodynamicModel,
        'simulation': generate_summed_simulated_fmri,
    },
}

VALID_KWARGS = [
    inspect.signature(model) for model in SEARCH_TYPES.values()
]
