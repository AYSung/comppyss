"""
CompPySS
Python implementation of the Comparative Proteomic Analysis Software Suite (CompPASS)
developed by Dr. Mathew Sowa for defining the human deubiquitinating enzyme interaction
landscape (Sowa, Mathew E., et al 2009). Based on the R packages CRomppass (David 
Nusinow)and SMAD (Qingzhou Zhang).
"""

from functools import partial
import math

import numpy as np
import pandas as pd


def _agg_max_spectral_count(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['prey', 'bait', 'replicate']).agg('max')


def entropy(s: pd.Series) -> float:
    """Calculates the Shannon entropy for a list of values. To avoid taking the log of
    zero, a fractional pseudocount of 1/(# of values) is added to each value.
    """
    p = (s + (1 / len(s))) / (s.sum() + 1)
    return sum(-p * np.log2(p))


def _calculate_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the following aggregate statistics from an input dataframe:

    ave_psm: mean of the PSM values for each bait-prey pair across replicates.
    p: number of replicates in which each bait-prey pair was detected.
    entropy: Shannon entropy across the replicates
    """
    return df.groupby(['bait', 'prey']).agg(
        ave_psm=('spectral_count', 'mean'),
        p=('spectral_count', 'count'),
        entropy=('spectral_count', entropy),
    )


def extend_series_with_zeroes(s: pd.Series, k: int) -> pd.Series:
    prey = s.index.get_level_values('prey')[0]
    n_to_extend = k - len(s)
    return s.append(pd.Series(0, index=((prey, i) for i in range(n_to_extend))))


def mean_(s: pd.Series, k: int) -> float:
    is_bait = s.index.get_level_values('bait') == s.index.get_level_values('prey')
    return s.loc[~is_bait].sum() / k


def std_(s: pd.Series, k: int) -> float:
    s = extend_series_with_zeroes(s, k)
    is_bait = _prey_equals_bait(s)
    mean = s.loc[~is_bait].sum() / k
    return math.sqrt(sum((s.loc[~is_bait] - mean) ** 2) / (k - 1))


def _prey_equals_bait(s: pd.Series) -> np.ndarray:
    return s.index.get_level_values('bait') == s.index.get_level_values('prey')


def _calculate_prey_stats(df: pd.DataFrame, k: int) -> pd.DataFrame:
    adjusted_mean = partial(mean_, k=k)
    adjusted_std = partial(std_, k=k)

    return df.groupby('prey').agg(
        prey_mean=('ave_psm', adjusted_mean),
        prey_std=('ave_psm', adjusted_std),
        f_sum=('ave_psm', 'count'),
    )


def z_score(s: pd.Series) -> float:
    return (s.ave_psm - s.prey_mean) / s.prey_std


def s_score(s: pd.Series, k: int) -> float:
    return math.sqrt((s.ave_psm * k) / s.f_sum)


def d_score(s: pd.Series, k: int) -> float:
    return math.sqrt(s.ave_psm * ((k / s.f_sum) ** s.p))


def wd_score(s: pd.Series, k: int) -> float:
    wd_inner = (k / s.f_sum) * (s.prey_std / s.prey_mean)
    return math.sqrt(s.ave_psm * (wd_inner**s.p))


def score_row(row: pd.Series, k: int) -> pd.DataFrame:
    z = z_score(row)
    s = s_score(row, k)
    d = d_score(row, k)
    wd = wd_score(row, k)

    return pd.Series(
        {
            'ave_psm': row.ave_psm,
            'z': z,
            's': s,
            'd': d,
            'wd': wd,
            'entropy': row.entropy,
        }
    )


def normalize_wd(s: pd.Series, normalization_factor=0.98) -> pd.Series:
    normalization_value = s.loc[s > 0].quantile(normalization_factor)
    return s / normalization_value


def comppass(input_df: pd.DataFrame) -> pd.DataFrame:
    k = input_df.bait.nunique()

    input_df = input_df.set_index(['bait', 'prey']).pipe(_agg_max_spectral_count)
    psm_stats = _calculate_aggregate_stats(input_df)
    prey_stats = _calculate_prey_stats(psm_stats, k=k)
    stats_table = psm_stats.join(prey_stats)

    result = (
        stats_table.apply(score_row, k=k, axis='columns')
        .assign(wd=lambda x: normalize_wd(x.wd))
        .reset_index()
    )

    return result
