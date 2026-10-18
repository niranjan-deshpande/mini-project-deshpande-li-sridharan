import numpy as np

from rhdecomp import ceiling_r2, cosine, disattenuated_cosine, null_span_r2, span_fit


def planted(d=128, k=4, res_norm=2.0, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(d, k))
    beta = rng.normal(size=k)
    r = rng.normal(size=d)
    r -= C @ np.linalg.lstsq(C, r, rcond=None)[0]  # make the residual orthogonal to span(C)
    r *= res_norm / np.linalg.norm(r)
    fitted = C @ beta
    return C, beta, r, fitted + r, fitted @ fitted / (fitted @ fitted + res_norm ** 2)


def test_span_fit_exact():
    C, beta, r, v, r2 = planted()
    fit = span_fit(C, v)
    assert np.allclose(fit.beta, beta)
    assert np.isclose(fit.r2, r2)
    assert np.allclose(fit.residual, r)
    assert np.allclose(C.T @ fit.residual, 0)
    assert np.allclose(fit.fitted + fit.residual, v)


def test_span_fit_noisy():
    C, beta, r, v, r2 = planted(d=512, res_norm=3.0, seed=1)
    rng = np.random.default_rng(2)
    fit = span_fit(C + 0.03 * rng.normal(size=C.shape), v + 0.03 * rng.normal(size=v.shape))
    assert np.allclose(fit.beta, beta, atol=0.05)
    assert abs(fit.r2 - r2) < 0.02


def test_cosine_helpers():
    assert np.isclose(cosine([1, 0], [1, 1]), 1 / np.sqrt(2))
    assert np.isclose(disattenuated_cosine(0.4, 0.8, 0.5), 0.4 / np.sqrt(0.4))
    assert np.isclose(ceiling_r2(0.3, 0.6), 0.5)


def test_null_span_r2_same_k():
    d, k = 64, 3
    rng = np.random.default_rng(3)
    v = rng.normal(size=d)
    pool = rng.normal(size=(6, d))
    exact = null_span_r2(v, pool, k, n_draws=1000)
    assert len(exact) == 20  # C(6, 3) <= n_draws: all subsets enumerated
    assert np.isclose(exact[0], span_fit(pool[:3].T, v).r2)
    # Sampled subsets from a large pool of random vectors: E[R^2] = k/d.
    sampled = null_span_r2(v, rng.normal(size=(300, d)), k, n_draws=500, seed=0)
    assert len(sampled) == 500
    assert abs(sampled.mean() - k / d) < 0.01
