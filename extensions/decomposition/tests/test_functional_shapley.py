import numpy as np
from sklearn.linear_model import LinearRegression

from rhdecomp import functional_r2, functional_r2_bootstrap, shapley_functional, shapley_geo, span_fit


def planted_tokens(n_roll=100, per=200, seed=0):
    """Candidate scores and v scores with rollout-level random effects; population R^2 = 0.6."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_roll), per)
    S = rng.normal(size=(len(g), 3)) + np.sqrt(0.5) * rng.normal(size=(n_roll, 3))[g]  # cov = 1.5 I
    beta = np.array([1.0, 0.5, -0.5])
    noise = rng.normal(size=len(g)) + np.sqrt(0.5) * rng.normal(size=n_roll)[g]   # var = 1.5
    return 2.0 + S @ beta + noise, S, g, 2.25 / (2.25 + 1.5)


def test_functional_r2_planted():
    y, S, g, truth = planted_tokens()
    r2 = functional_r2(y, S)
    assert np.isclose(r2, LinearRegression().fit(S, y).score(S, y))
    assert abs(r2 - truth) < 0.04
    boot = functional_r2_bootstrap(y, S, g, n_boot=300, seed=0)
    assert np.isclose(boot["r2"], r2)
    lo, hi = boot["ci"]
    assert lo < truth < hi and hi - lo < 0.2


def test_shapley_additivity():
    rng = np.random.default_rng(1)
    C = rng.normal(size=(64, 5)) + rng.normal(size=(64, 1))  # correlated columns
    v = C @ rng.normal(size=5) + rng.normal(size=64)
    geo = shapley_geo(C, v)
    assert np.isclose(geo["shapley"].sum(), geo["full_r2"])
    assert np.isclose(geo["full_r2"], span_fit(C, v).r2)

    S = rng.normal(size=(2000, 5)) + rng.normal(size=(2000, 1))
    y = S @ rng.normal(size=5) + rng.normal(size=2000)
    fun = shapley_functional(y, S)
    assert np.isclose(fun["shapley"].sum(), fun["full_r2"])
    assert np.isclose(fun["full_r2"], functional_r2(y, S))


def test_shapley_orthogonal_equals_individual():
    rng = np.random.default_rng(2)
    C = np.linalg.qr(rng.normal(size=(64, 4)))[0] * np.array([1, 2, 3, 4])
    v = rng.normal(size=64)
    geo = shapley_geo(C, v)
    assert np.allclose(geo["shapley"], geo["single_r2"])
    assert np.allclose(geo["unique"], geo["single_r2"])

    S = rng.normal(size=(500, 4)); S -= S.mean(0)
    S = np.linalg.qr(S)[0]  # orthonormal, still centered -> uncorrelated scores
    y = S @ np.array([1.0, -2.0, 0.5, 0.0]) * 10 + rng.normal(size=500)
    fun = shapley_functional(y, S)
    assert np.allclose(fun["shapley"], fun["single_r2"])
    assert np.allclose(fun["unique"], fun["single_r2"])


def test_shapley_duplicate_splits_equally():
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.normal(size=(64, 2)))[0]
    a, b = Q[:, 0], Q[:, 1]
    v = 2 * a + b + 0.5 * rng.normal(size=64)
    single = shapley_geo(np.column_stack([a, b]), v)
    dup = shapley_geo(np.column_stack([a, a, b]), v)
    assert np.isclose(dup["shapley"][0], dup["shapley"][1])
    assert np.isclose(dup["shapley"][0], single["single_r2"][0] / 2)
    assert np.isclose(dup["shapley"][2], single["single_r2"][1])
    assert np.allclose(dup["unique"][:2], 0)  # either copy alone carries everything
    assert np.isclose(dup["shapley"].sum(), single["full_r2"])

    S = rng.normal(size=(1000, 2))
    y = S @ [1.0, 0.5] + rng.normal(size=1000)
    fun = shapley_functional(y, S[:, [0, 0, 1]])
    assert np.isclose(fun["shapley"][0], fun["shapley"][1])
    assert np.isclose(fun["shapley"].sum(), functional_r2(y, S))
