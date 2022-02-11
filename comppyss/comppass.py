"""
CompPySS
Python implementation of the Comparative Proteomic Analysis Software Suite (CompPASS)
developed by Dr. Mathew Sowa for defining the human deubiquitinating enzyme interaction
landscape (Sowa, Mathew E., et al 2009). Based on the R packages cRomppass (David
Nusinow) and SMAD (Qingzhou Zhang).
"""

from functools import partial
import math

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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


def calculate_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
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
    is_bait = prey_equals_bait(s)
    return s.loc[~is_bait].sum() / n


def extend_series_with_zeroes(s: pd.Series, n: int) -> pd.Series:
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
    s = extend_series_with_zeroes(s, n)
    is_bait = prey_equals_bait(s)
    mean = s.loc[~is_bait].sum() / n
    return math.sqrt(sum((s.loc[~is_bait] - mean) ** 2) / (n - 1))


def prey_equals_bait(s: pd.Series) -> np.ndarray:
    """Return an indexer for elements of a series where the prey is the same as the
    bait.
    """
    return s.index.get_level_values('bait') == s.index.get_level_values('prey')


def calculate_prey_stats(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Calculate the following statistics for each prey in the dataset:

    prey_mean: mean of the ave_psm, see mean_ function documentation for details.
    prey_std: sample standard deviation of the ave_psm, see std_ function documentation
        for details.
    f_sum: number of baits in which the prey was observed.
    """
    # since agg() can only take functions with a single argument, the number of unique
    # baits n needs to be pre-loaded into the calculation.
    adjusted_mean = partial(mean_, n=n)
    adjusted_std = partial(std_, n=n)

    return df.groupby('prey').agg(
        prey_mean=('ave_psm', adjusted_mean),
        prey_std=('ave_psm', adjusted_std),
        f_sum=('ave_psm', 'count'),
    )


def z_score(df: pd.DataFrame) -> pd.Series:
    return (df.ave_psm - df.prey_mean) / df.prey_std


def s_score(df: pd.DataFrame, n: int) -> pd.Series:
    return np.sqrt((df.ave_psm * n) / df.f_sum)


def d_score(df: pd.DataFrame, n: int) -> pd.Series:
    return np.sqrt(df.ave_psm * ((n / df.f_sum) ** df.p))


def wd_score(df: pd.DataFrame, n: int) -> pd.Series:
    wd_inner = (n / df.f_sum) * (df.prey_std / df.prey_mean)
    return normalize_wd(np.sqrt(df.ave_psm * (wd_inner**df.p)))


def normalize_wd(s: pd.Series, normalization_factor=0.98) -> pd.Series:
    """Normalize WD scores to the 98%ile of non-zero values, such that only ~2% of WD
    scores are >1."""
    normalization_value = s.loc[s > 0].quantile(normalization_factor)
    return s / normalization_value


def _calculate_scores(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Calculates Z, S, D, and WD scores from a DataFrame with the following columns:
    'ave_psm', 'p', 'prey_mean', 'prey_std', and 'f_sum'.
    """
    return df.assign(
        z=z_score,
        s=partial(s_score, n=n),
        d=partial(d_score, n=n),
        wd=partial(wd_score, n=n),
    )


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

    input_df = preprocess(input_df)
    psm_stats = calculate_aggregate_stats(input_df)
    prey_stats = calculate_prey_stats(psm_stats, n)
    stats_table = psm_stats.join(prey_stats)

    result = stats_table.pipe(_calculate_scores).reset_index()

    return result[['bait', 'prey', 'ave_psm', 'z', 'wd', 'entropy']]
