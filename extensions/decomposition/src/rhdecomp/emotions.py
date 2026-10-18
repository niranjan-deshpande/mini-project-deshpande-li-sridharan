"""Phase 5 emotion regression: penalized regression of token-level v scores on standardized emotion scores.

sklearn's objectives are scaled per sample: Lasso minimizes (1/2T)||r||^2 + alpha ||b||_1 and Ridge minimizes
||r||^2 + alpha ||b||^2, so the brief's lambda equals 2T*alpha for the lasso and alpha for ridge. Emotion scores
are standardized inside the pipeline, so the scaler is refit on each CV training fold.
"""
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = {"lasso": np.logspace(-4, 0, 17), "ridge": np.logspace(-2, 5, 15)}


def _model(kind, alpha=1.0):
    est = Lasso(alpha=alpha, max_iter=20000, precompute=True) if kind == "lasso" else Ridge(alpha=alpha)
    return make_pipeline(StandardScaler(), est)


def fit_penalized(y, E, groups, train, test, kind="lasso", alphas=None, n_folds=5):
    """Choose alpha by GroupKFold CV over rollouts on `train`, refit on all of `train`, score R^2 on `test`.

    y: [T] v token scores, E: [T, m] emotion token scores, groups: [T] rollout ids, train/test: [T] masks
    (split by task, so no rollout is in both). Coefficients are per standard deviation of each emotion score.
    """
    y, E, groups = np.asarray(y, float), np.asarray(E, float), np.asarray(groups)
    train, test = np.asarray(train, bool), np.asarray(test, bool)
    alphas = DEFAULT_ALPHAS[kind] if alphas is None else alphas
    search = GridSearchCV(_model(kind), {f"{kind}__alpha": list(alphas)}, cv=GroupKFold(n_folds), scoring="r2")
    search.fit(E[train], y[train], groups=groups[train])
    best = search.best_estimator_
    est = best[-1]
    return {"kind": kind, "alpha": float(search.best_params_[f"{kind}__alpha"]),
            "cv_r2": float(search.best_score_),
            "cv_curve": (np.asarray(alphas), search.cv_results_["mean_test_score"]),
            "heldout_r2": float(r2_score(y[test], best.predict(E[test]))),
            "coef": est.coef_.copy(), "intercept": float(est.intercept_), "model": best}


def selection_frequencies(y, E, groups, alpha, n_boot=200, seed=0):
    """Refit the lasso at a fixed alpha on rollout-bootstrap resamples; fraction of fits where each emotion is nonzero.

    alpha is held at the CV choice from the full fit (re-running CV inside each resample would cost n_folds x more
    and duplicated rollouts would straddle folds). Claims should be made at the level of correlated clusters.
    """
    y, E, groups = np.asarray(y, float), np.asarray(E, float), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    coefs = np.empty((n_boot, E.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        coefs[b] = _model("lasso", alpha).fit(E[idx], y[idx])[-1].coef_
    return {"freq": (coefs != 0).mean(0), "coef_mean": coefs.mean(0), "coef_sd": coefs.std(0), "coefs": coefs}
