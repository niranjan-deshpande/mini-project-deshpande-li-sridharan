import hashlib

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from rhdecomp import auroc, aurocs, length_residualize, paired_task_bootstrap, split_masks, task_half


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(5, 300)
        y = rng.random(n) < rng.uniform(0.1, 0.9)
        if y.all() or not y.any():
            continue
        s = rng.integers(0, 5, n) + 0.5 * y  # heavy ties
        assert np.isclose(auroc(y, s), roc_auc_score(y, s))
    assert np.isnan(auroc([1, 1], [0.2, 0.3]))


def test_split_rule_known_strings():
    known = {"django__django-11099": 0, "sympy__sympy-20590": 0, "astropy__astropy-12907": 0,
             "task_0": 1, "abc": 0}
    for s, h in known.items():
        assert task_half(s) == h == int(hashlib.md5(s.encode()).hexdigest(), 16) % 2
    sel, held = split_masks(list(known))
    assert held.tolist() == [False, False, False, True, False] and (sel == ~held).all()


def simulate(rng, n_tasks=60, per=10, delta=1.0, tau=np.sqrt(0.5)):
    tasks = np.repeat(np.arange(n_tasks), per)
    y = rng.random(len(tasks)) < 0.3
    s = delta * y + tau * rng.normal(size=n_tasks)[tasks] + rng.normal(size=len(tasks))
    return y, s, tasks


def test_bootstrap_ci_coverage():
    # Population AUROC with a shared task effect: Phi(delta / sqrt(2 (1 + tau^2))).
    truth = norm.cdf(1.0 / np.sqrt(2 * 1.5))
    rng = np.random.default_rng(1)
    hits = 0
    for i in range(100):
        y, s, tasks = simulate(rng)
        lo, hi = paired_task_bootstrap({"p": s}, y, tasks, n_boot=200, seed=i)["ci"]["p"]
        hits += lo <= truth <= hi
    assert hits >= 85


def test_paired_difference():
    rng = np.random.default_rng(2)
    y, s, tasks = simulate(rng, n_tasks=80)
    weak = 0.3 * y + rng.normal(size=len(y))
    res = paired_task_bootstrap({"strong": s + y, "weak": weak, "copy": s + y}, y, tasks, n_boot=500)
    assert res["diff"][("strong", "weak")]["ci"][0] > 0
    assert res["diff"][("strong", "copy")]["ci"] == (0.0, 0.0)  # identical probes: paired diff is exactly 0
    assert res["n_skipped"] == 0 and res["samples"].shape == (500, 3)
    assert np.isclose(res["point"]["strong"], auroc(y, s + y))


def test_bootstrap_skips_single_class_resamples():
    tasks = np.repeat(["a", "b", "c", "d"], 5)
    y = np.zeros(20, bool); y[:2] = True  # only task "a" has positives
    s = np.arange(20.0)
    res = paired_task_bootstrap({"p": s}, y, tasks, n_boot=200, seed=0)
    assert 0 < res["n_skipped"] < 200
    assert len(res["samples"]) == 200 - res["n_skipped"]
    assert np.all(np.isfinite(res["ci"]["p"]))


def test_length_control():
    rng = np.random.default_rng(3)
    n_tok = np.exp(rng.normal(5, 1, 4000))
    y = np.log(n_tok) + 0.5 * rng.normal(size=4000) > 5.8
    scores = {"length_only": 3 * np.log(n_tok) + rng.normal(size=4000),
              "length_plus_signal": np.log(n_tok) + 1.5 * y + 0.5 * rng.normal(size=4000)}
    raw = aurocs(scores, y)
    ctrl = aurocs(length_residualize(scores, n_tok), y)
    assert raw["length_only"] > 0.8
    assert abs(ctrl["length_only"] - 0.5) < 0.04
    assert ctrl["length_plus_signal"] > 0.6
