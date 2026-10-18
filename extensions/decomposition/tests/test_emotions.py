import numpy as np

from rhdecomp import fit_penalized, selection_frequencies, split_masks


def test_emotion_regression_planted_sparse():
    rng = np.random.default_rng(0)
    n_roll, per, m = 80, 50, 12
    g = np.repeat(np.arange(n_roll), per)
    E = rng.normal(size=(len(g), m)) + 0.5 * rng.normal(size=(n_roll, m))[g]
    beta = np.zeros(m); beta[[0, 3]] = [1.0, -0.7]
    y = E @ beta + rng.normal(size=len(g))
    truth = beta @ (1.25 * beta) / (beta @ (1.25 * beta) + 1)  # cov(E) = 1.25 I

    sel, held = split_masks([f"task{r}" for r in g])
    lasso = fit_penalized(y, E, g, sel, held, kind="lasso", n_folds=4)
    ridge = fit_penalized(y, E, g, sel, held, kind="ridge", n_folds=4)
    for fit in (lasso, ridge):
        assert abs(fit["heldout_r2"] - truth) < 0.06
    assert set(np.flatnonzero(np.abs(lasso["coef"]) > 0.1)) == {0, 3}

    freq = selection_frequencies(y[sel], E[sel], g[sel], alpha=lasso["alpha"], n_boot=30)["freq"]
    assert freq[0] == 1 and freq[3] == 1
    assert np.delete(freq, [0, 3]).mean() < freq[[0, 3]].mean()
