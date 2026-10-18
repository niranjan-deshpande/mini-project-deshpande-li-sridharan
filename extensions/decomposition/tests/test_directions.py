import numpy as np

from rhdecomp import cosine, dom, label_shuffled_doms, split_half_reliability


def make_pairs(n, d, mu, sigma, rng):
    """Pairs sharing a per-pair base (cancels in the DoM); A adds the true direction mu."""
    base = rng.normal(size=(n, d))
    A = base + mu + sigma * rng.normal(size=(n, d))
    B = base + sigma * rng.normal(size=(n, d))
    return A, B


def test_dom_and_idx():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(10, 4)), rng.normal(size=(10, 4))
    assert np.allclose(dom(A, B), A.mean(0) - B.mean(0))
    assert np.allclose(dom(A, B, [1, 3]), A[[1, 3]].mean(0) - B[[1, 3]].mean(0))


def test_split_half_reliability_matches_analytic():
    # Each pair difference = mu + noise with per-coordinate variance 2 sigma^2, so a DoM over m pairs has
    # noise norm^2 ~ 2 sigma^2 d / m and the reliability is n / (n + c) with c = 2 sigma^2 d / ||mu||^2.
    d, c = 256, 100.0
    mu = np.zeros(d); mu[0] = 1.0
    sigma = np.sqrt(c / (2 * d))
    rng = np.random.default_rng(1)
    rhos = []
    for n in (50, 100, 200, 400):
        # rho-hat varies across datasets (sd ~0.05 at n = 50), so test its mean over replicate datasets.
        rels = [split_half_reliability(*make_pairs(n, d, mu, sigma, rng), n_splits=100, seed=i) for i in range(8)]
        rho = np.mean([r.rho for r in rels])
        assert abs(rho - n / (n + c)) < 0.04, (n, rho, n / (n + c))
        assert len(rels[0].r_samples) == 100 and np.isclose(rels[0].r, rels[0].r_samples.mean())
        rhos.append(rho)
    assert np.all(np.diff(rhos) > 0)


def test_label_shuffled_doms_balanced_flips_do_not_leak_v():
    # Strong shared direction relative to per-pair noise (the regime seen on real data).
    d, n = 256, 135
    rng = np.random.default_rng(2)
    mu = rng.normal(size=d); mu *= 10 / np.linalg.norm(mu)
    A, B = make_pairs(n, d, mu, sigma=10 / np.sqrt(2 * d), rng=rng)
    nulls = label_shuffled_doms(A, B, k=200, seed=0)
    assert nulls.shape == (200, d)
    balanced = np.mean([abs(cosine(z, mu)) for z in nulls])
    assert balanced < 2 / np.sqrt(d)  # pure-noise level is ~0.8/sqrt(d)

    # Independent coin flips leak mu through the sum(eps) imbalance.
    D = A - B
    coin = np.mean([abs(cosine(rng.choice([-1, 1], n) @ D / n, mu)) for _ in range(200)])
    assert coin > 0.3

    # Exactly floor(n/2) pairs flipped: null DoM = (eps @ D) / n with sum(eps) = n mod 2.
    nulls_even = label_shuffled_doms(A[:134], B[:134], k=5, seed=1)
    D_even = D[:134]
    for z in nulls_even:
        eps, *_ = np.linalg.lstsq(D_even.T, z * 134, rcond=None)
        assert np.allclose(np.abs(eps), 1, atol=1e-6) and abs(np.round(eps).sum()) == 0
