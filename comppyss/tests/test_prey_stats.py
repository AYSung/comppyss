import math

import pytest

from comppyss.comppass import *


def test_prey_equals_bait(prey_series_1, prey_series_2, prey_series_3):
    assert prey_equals_bait(prey_series_2).sum() == 1
    assert prey_equals_bait(prey_series_3).sum() == 0
    assert prey_equals_bait(prey_series_1).sum() == 1


def test_extend_series(prey_series_1, prey_series_2, prey_series_3):
    extended_series_1 = extend_series_with_zeroes(prey_series_1, 12)
    extended_series_2 = extend_series_with_zeroes(prey_series_2, 10)
    extended_series_3 = extend_series_with_zeroes(prey_series_3, 12)

    assert len(extended_series_1) == 12
    assert len(extended_series_2) == 10
    assert len(extended_series_3) == 12

    assert extended_series_1.sum() == prey_series_1.sum()
    assert extended_series_2.sum() == prey_series_2.sum()
    assert extended_series_3.sum() == prey_series_3.sum()

    assert (
        prey_equals_bait(extended_series_1).sum()
        == prey_equals_bait(prey_series_1).sum()
    )
    assert (
        prey_equals_bait(extended_series_2).sum()
        == prey_equals_bait(prey_series_2).sum()
    )
    assert (
        prey_equals_bait(extended_series_3).sum()
        == prey_equals_bait(prey_series_3).sum()
    )


def test_adjusted_mean(prey_series_1, prey_series_2, prey_series_3):
    assert mean_(prey_series_1, 12) == 12
    assert mean_(prey_series_2, 10) == 1
    assert mean_(prey_series_3, 12) == 1


def test_adjusted_std(prey_series_1, prey_series_2, prey_series_3):
    assert math.isclose(std_(prey_series_1, 12), math.sqrt(974 / 11))
    assert math.isclose(std_(prey_series_2, 10), math.sqrt(11 / 9))
    assert math.isclose(std_(prey_series_3, 12), math.sqrt(24 / 11))


@pytest.fixture
def prey_series_1():
    index = pd.MultiIndex.from_tuples(
        (('A', x) for x in 'ABCDEFGHIJKL'), names=['prey', 'bait']
    )
    return pd.Series([460, 13, 13, 38, 7, 4, 15, 22, 9, 10, 3, 10], index=index)


@pytest.fixture
def prey_series_2():
    index = pd.MultiIndex.from_tuples(
        (('B', x) for x in 'ABCDEF'), names=['prey', 'bait']
    )
    return pd.Series([2, 46, 3, 2, 2, 1], index=index)


@pytest.fixture
def prey_series_3():
    index = pd.MultiIndex.from_tuples(
        (('Z', x) for x in 'ABCDEF'), names=['prey', 'bait']
    )
    return pd.Series([2, 2, 1, 1, 5, 1], index=index)


@pytest.fixture
def n_baits():
    return 12
