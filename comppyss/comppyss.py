import math

import numpy as np
import pandas as pd


def entropy(s: pd.Series) -> float:
    p = (s + (1 / len(s))) / (s.sum() + 1)
    return sum(-p * np.log2(p))


def z_score(s: pd.Series) -> float:
    return (s.ave_psm - s.prey_mean) / s.prey_sd


def s_score(s: pd.Series, k: int) -> float:
    return math.sqrt((s.ave_psm * k) / s.f_sum)


def d_score(s: pd.Series, k: int) -> float:
    return math.sqrt(s.ave_psm * ((k / s.f_sum) ** s.p))


def wd_score(s: pd.Series, k: int) -> float:
    wd_inner = (k / s.f_sum) * (s.prey_sd / s.prey_mean)
    return math.sqrt(s.ave_psm * (wd_inner**s.p))


def normalize_wd(s: pd.Series, normalization_factor=0.98) -> pd.Series:
    normalization_value = s.loc[s > 0].quantile(normalization_factor)
    return s / normalization_value


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


def comppass(df: pd.DataFrame) -> pd.DataFrame:
    # columns = ['bait', 'prey', 'replicate', 'spectral_count']
    # check df is correct format

    baits = set(df.bait.unique())
    k = len(baits)

    stats = _calculate_aggregate_stats(df)

    f_sum = stats.groupby('prey').bait.count().rename('f_sum')

    def extend_series_with_zeroes(s: pd.Series, k: int) -> pd.Series:
        if len(s) == k:
            return s
        elif len(s) < k:
            return s.append(pd.Series(0, index=range(k - len(s))))
        else:
            raise ValueError(
                'Series length cannot be greater than number of unique baits in dataset.'
            )

    def adjusted_mean(s: pd.Series) -> float:
        return extend_series_with_zeroes(s, k).mean()

    def adjusted_std(s: pd.Series) -> float:
        return extend_series_with_zeroes(s, k).std()

    def adjusted_std_bait(s: pd.Series) -> float:
        if len(s) == k:
            return bait_std(s)
        else:
            return bait_std(extend_series_with_zeroes(s, k - 1))

    def bait_std(s: pd.Series) -> float:
        return math.sqrt(sum((s - adjusted_mean(s)) ** 2) / (k - 1))

    non_bait_prey_stats = (
        stats.loc[~stats.prey.isin(baits)]
        .groupby('prey')
        .agg(prey_mean=('ave_psm', adjusted_mean), prey_sd=('ave_psm', adjusted_std))
    )
    bait_prey_stats = (
        stats.loc[stats.prey.isin(baits) & (stats.bait != stats.prey)]
        .groupby('prey')
        .agg(
            prey_mean=('ave_psm', adjusted_mean), prey_sd=('ave_psm', adjusted_std_bait)
        )
    )
    prey_stats = pd.concat((non_bait_prey_stats, bait_prey_stats))
    new_stats = stats.set_index(['bait', 'prey']).join(prey_stats).join(f_sum)

    result = new_stats.apply(score_row, axis='columns').assign(
        wd=lambda x: normalize_wd(x.wd)
    )[['ave_psm', 'z', 'wd', 'entropy']]

    return result


def _calculate_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(['bait', 'prey'])
        .agg(
            ave_psm=('spectral_count', 'mean'),
            p=('spectral_count', 'count'),
            entropy=('spectral_count', entropy),
        )
        .reset_index()
    )
