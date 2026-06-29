import numpy as np
import pandas as pd
import pytest

from aipyw.benchmarks import (
    ACIC2016_N_REPLICATIONS,
    ACIC2016_N_SETTINGS,
    ACIC2017_N_REPLICATIONS,
    ACIC2017_N_SETTINGS,
    iter_acic2016,
    iter_acic2017,
    load_acic2016_input,
    load_acic2016_parameters,
    load_acic2017_input,
    load_acic2017_parameters,
    simulate_acic2016,
    simulate_acic2017,
)


def test_acic_data_primitives_have_expected_shapes():
    assert load_acic2016_input().shape == (4802, 58)
    assert load_acic2016_parameters().shape == (ACIC2016_N_SETTINGS, 6)
    assert load_acic2017_input().shape == (4302, 58)
    assert load_acic2017_parameters().shape == (ACIC2017_N_SETTINGS, 4)


def test_acic2016_sample_contains_aipyw_arrays_and_truth():
    sample = simulate_acic2016(1, 1, random_seed=123)
    y, a, X = sample.as_tuple()

    assert y.shape == a.shape == (4802,)
    assert X.shape[0] == 4802
    assert set(np.unique(a)) <= {0, 1}
    assert sample.e.shape == (4802,)
    assert np.all((sample.e >= 0.01) & (sample.e <= 0.99))
    assert np.isclose(sample.true_ate, np.mean(sample.mu1 - sample.mu0))
    assert np.isclose(sample.true_att, np.mean((sample.mu1 - sample.mu0)[a == 1]))


def test_acic2016_is_stable_for_grid_indices():
    first = simulate_acic2016(ACIC2016_N_SETTINGS, ACIC2016_N_REPLICATIONS)
    again = simulate_acic2016(ACIC2016_N_SETTINGS, ACIC2016_N_REPLICATIONS)
    pd.testing.assert_frame_equal(first.x, again.x)
    np.testing.assert_allclose(first.y, again.y)
    np.testing.assert_array_equal(first.a, again.a)

    with pytest.raises(ValueError):
        simulate_acic2016(ACIC2016_N_SETTINGS + 1, 1)
    with pytest.raises(ValueError):
        simulate_acic2016(1, ACIC2016_N_REPLICATIONS + 1)


def test_acic2017_sample_contains_cate_truth():
    sample = simulate_acic2017(1, 1, random_seed=123)
    y, a, X = sample.as_tuple()

    assert y.shape == a.shape == sample.alpha.shape == (4302,)
    assert X.shape[0] == 4302
    assert set(np.unique(a)) <= {0, 1}
    assert np.isclose(sample.true_ate, np.mean(sample.alpha))
    assert np.isclose(sample.true_att, np.mean(sample.alpha[a == 1]))


def test_acic_iterators_accept_subsets():
    acic16 = list(iter_acic2016(settings=[1, 2], replications=[1]))
    acic17 = list(iter_acic2017(settings=[1], replications=[1, 2, 3]))

    assert [(s.setting, s.replication) for s in acic16] == [(1, 1), (2, 1)]
    assert [(s.setting, s.replication) for s in acic17] == [(1, 1), (1, 2), (1, 3)]
