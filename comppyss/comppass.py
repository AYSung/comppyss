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


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Sets DataFrame index, drops rows with spectral_count of 0, and only keep the
    maximum spectral count for bait-prey pairs if there are duplicates within a given
    replicate.
    """
    return (
        df.set_index(['bait', 'prey'])
        .loc[lambda x: x.spectral_count > 0]
        .groupby(['prey', 'bait', 'replicate'])
        .agg('max')
    )


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
    entropy: Shannon entropy.
    """
    return df.groupby(['bait', 'prey']).agg(
        ave_psm=('spectral_count', 'mean'),
        p=('spectral_count', 'count'),
        entropy=('spectral_count', entropy),
    )


def mean_(s: pd.Series, n: int) -> float:
    """Calculates the mean of a series. Denominator is the total number of unique baits,
    so ave_psm for bait-prey pairs that were not detected are counted as zeroes."""
    is_bait = s.index.get_level_values('bait') == s.index.get_level_values('prey')
    return s.loc[~is_bait].sum() / n


def _extend_series_with_zeroes(s: pd.Series, n: int) -> pd.Series:
    """Extends Series to n elements by filling in missing elements with zero values.
    'prey' level of the multindex is preserved. This is used for calculating the
    standard deviation.
    """
    prey = s.index.get_level_values('prey')[0]
    n_to_extend = n - len(s)
    return s.append(pd.Series(0, index=((prey, None) for _ in range(n_to_extend))))


def std_(s: pd.Series, n: int) -> float:
    """Calculate the sample standard deviation. Values for bait-prey pairs that were
    not detected are set to 0 for the standard deviation calculation. Excludes values
    where the bait and the prey are the same.
    """
    s = _extend_series_with_zeroes(s, n)
    is_bait = _prey_equals_bait(s)
    mean = s.loc[~is_bait].sum() / n
    return math.sqrt(sum((s.loc[~is_bait] - mean) ** 2) / (n - 1))


def _prey_equals_bait(s: pd.Series) -> np.ndarray:
    """Return an indexer for elements of a series where the prey is the same as the
    bait.
    """
    return s.index.get_level_values('bait') == s.index.get_level_values('prey')


def _calculate_prey_stats(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Calculate the following statistics for each prey in the dataset:

    prey_mean: mean of the ave_psm, see mean_ function documentation for details.
    prey_std: sample standard deviation of the ave_psm, see std_ function documentation
        for details.
    f_sum: number of baits in which the prey was observed.
    """
    # since agg() can only take functions with a single argument, the number of unique
    # baits n needs to be pre-loaded into the calculation.
    adjusted_mean = partial(mean_, n)
    adjusted_std = partial(std_, n)

    return df.groupby('prey').agg(
        prey_mean=('ave_psm', adjusted_mean),
        prey_std=('ave_psm', adjusted_std),
        f_sum=('ave_psm', 'count'),
    )


def z_score(s: pd.Series) -> float:
    return (s.ave_psm - s.prey_mean) / s.prey_std


def s_score(s: pd.Series, n: int) -> float:
    return math.sqrt((s.ave_psm * n) / s.f_sum)


def d_score(s: pd.Series, n: int) -> float:
    return math.sqrt(s.ave_psm * ((n / s.f_sum) ** s.p))


def wd_score(s: pd.Series, n: int) -> float:
    wd_inner = (n / s.f_sum) * (s.prey_std / s.prey_mean)
    return math.sqrt(s.ave_psm * (wd_inner**s.p))


def score_row(row: pd.Series, n: int) -> pd.DataFrame:
    """Calculates Z, S, D, and WD scores for a given a row with the following columns:
    'ave_psm', 'p', 'prey_mean', 'prey_std', and 'f_sum'.
    """
    z = z_score(row)
    s = s_score(row, n)
    d = d_score(row, n)
    wd = wd_score(row, n)

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
    """Normalize WD scores to the 98%ile of non-zero values, such that only ~2% of WD
    scores are >1."""
    normalization_value = s.loc[s > 0].quantile(normalization_factor)
    return s / normalization_value


def comppass(input_df: pd.DataFrame) -> pd.DataFrame:
    """Perform CompPASS analysis on input DataFrame. Input DataFrame must have the
    following columns:

    'prey': gene name or other identifier for each bait.
    'bait': gene name or other identifier for each bait.
    'replicate': id for replicates of a single bait (e.g. 1, 2, 3 or A, B, C)
    'spectral_count': peptide spectral matches for each bait-prey pair.

    NOTE: Both the prey and bait columns should use the same system of identifiers
    in order to be able to properly handle instances where the bait and the prey
    are the same. The spectral_count column should not contain any rows where the
    spectral count is zero.
    """
    # n: number of unique baits in the dataset
    n = input_df.bait.nunique()

    input_df = _preprocess(input_df)
    psm_stats = _calculate_aggregate_stats(input_df)
    prey_stats = _calculate_prey_stats(psm_stats, n)
    stats_table = psm_stats.join(prey_stats)

    result = (
        stats_table.apply(score_row, n, axis='columns')
        .assign(wd=lambda x: normalize_wd(x.wd))
        .reset_index()
    )

    return result
