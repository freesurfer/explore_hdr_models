from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd

from feeg_fmri_sync.constants import STATISTICS


def get_quantile_by_model_name(df: pd.DataFrame, quantile: float = .1,
                               columns_to_quantile: Optional[Iterable] = None) -> pd.DataFrame:
    if not columns_to_quantile:
        columns_to_quantile = ['beta', 'residual_variance', 'correlation_coefficient']

    data = {column: [] for column in df.columns}
    data['quantile_variable'] = []

    for model_name in df['model_name'].unique():
        df_for_model = df[df['model_name'] == model_name]
        quantile_values = df_for_model[['voxel_name'] + list(STATISTICS)].groupby('voxel_name').quantile(quantile)
        for column in columns_to_quantile:
            for voxel_name, row in quantile_values.iterrows():
                mask = np.all([
                    (df_for_model[column] <= row[column]).tolist(),
                    (df_for_model['voxel_name'] == voxel_name).tolist()
                ], axis=0)
                data_list = df_for_model[mask]
                for out_stat in df_for_model.columns:
                    data[out_stat].extend(data_list[out_stat].tolist())
                data['quantile_variable'].extend([column for _ in range(len(data_list.index))])

    return pd.DataFrame(data)


def get_descriptions_by_model_name(df, variables_to_filter_on_with_option: Iterable[Tuple[str, str]]):
    for variable_to_filter_on, filter_option in variables_to_filter_on_with_option:
        if variable_to_filter_on not in STATISTICS:
            raise ValueError(f"cannot filter on variable not in {STATISTICS}")
        if filter_option not in ['min', 'max']:
            raise NotImplementedError(f'Only filter options supported are "min" and "max". Unable to filter on'
                                      f' {filter_option}')
    descriptions = []
    for model_name in df['model_name'].unique():
        df_for_model = df[df['model_name'] == model_name]
        grouped_by_voxel = df_for_model[['voxel_name'] + list(STATISTICS)].groupby('voxel_name')
        description = grouped_by_voxel.describe()
        for variable_to_filter_on, filter_option in variables_to_filter_on_with_option:
            if filter_option == 'min':
                index_for_best_dta = grouped_by_voxel.idxmin()[variable_to_filter_on]
            else:
                index_for_best_dta = grouped_by_voxel.idxmax()[variable_to_filter_on]
            best_dta = df_for_model.loc[index_for_best_dta, ['delta', 'tau', 'alpha']].set_index(index_for_best_dta.index)
            columns = pd.MultiIndex.from_arrays([[variable_to_filter_on for _ in range(3)], ['delta', 'tau', 'alpha']])
            description[columns] = best_dta
        description.name = f'{model_name}'
        descriptions.append(description)
    return descriptions

