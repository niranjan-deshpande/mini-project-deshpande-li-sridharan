import numpy as np

from rhdecomp import (add_decomposition, compare_max_detector, peak_decomposition, pool_max, project,
                      project_rollouts, span_fit, split_masks, summarize_peaks)


def test_pool_max_by_hand():
    s = {"a": np.array([1.0, 5.0, 2.0, -1.0, 3.0])}
    p = np.array([0, 0, 2, 2, -1])  # passage 1 has no tokens, last token is outside any passage
    out = pool_max(s, p, 3)["a"]
    assert out[0] == 5 and np.isnan(out[1]) and out[2] == 2


def test_decomposition_scores_pooled_from_tokens():
    rng = np.random.default_rng(0)
    d, k = 32, 3
    C = rng.normal(size=(d, k))
    v = C @ rng.normal(size=k) + rng.normal(size=d)
    fit = span_fit(C, v)
    dirs = {"v": v, **{f"c{i}": C[:, i] for i in range(k)}}
    rollouts = [(rng.normal(size=(40, d)).astype(np.float16), np.repeat([0, 1, 2, 3], 10)),
                (rng.normal(size=(30, d)).astype(np.float16), np.repeat([4, 5, -1], 10))]
    tok, pidx, ridx = project_rollouts(rollouts, dirs)
    assert len(pidx) == len(ridx) == 70 and ridx[-1] == 1
    tok = add_decomposition(tok, "v", [f"c{i}" for i in range(k)], fit.beta)

    X = np.concatenate([np.asarray(x, float) for x, _ in rollouts])
    direct = project(X, {"fit": fit.fitted, "res": fit.residual})
    assert np.allclose(tok["fit"], direct["fit"], atol=1e-3)
    assert np.allclose(tok["res"], direct["res"], atol=1e-3)

    P = pool_max(tok, pidx, 6)
    # Max-pooling is nonlinear: combining per-candidate passage maxes gives a different (wrong) answer.
    combined = sum(b * P[f"c{i}"] for i, b in enumerate(fit.beta))
    assert not np.allclose(P["fit"], combined)


def mixture_data(mixture, n=3000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    tasks = np.array([f"task{t}" for t in rng.integers(0, 300, n)])
    y = rng.random(n) < 0.3
    S = rng.normal(size=(n, k))
    if mixture:  # each positive elevated on exactly one candidate
        S[np.flatnonzero(y), rng.integers(0, k, y.sum())] += 3.0
    else:        # additive: every candidate slightly elevated
        S[y] += 1.0
    scores = {f"c{i}": S[:, i] for i in range(k)}
    scores["fit"] = S.sum(1)  # the linear combination C beta with beta = 1
    return scores, y, tasks, [f"c{i}" for i in range(k)]


def test_max_detector_wins_on_mixture_only():
    for mixture in (True, False):
        scores, y, tasks, names = mixture_data(mixture)
        sel, held = split_masks(tasks)
        res = compare_max_detector(scores, y, tasks, names, "fit", sel, held, n_boot=300)
        lo, hi = res["diff"][("max", "fit")]["ci"]
        if mixture:
            assert res["point"]["max"] > res["point"]["fit"] + 0.05 and lo > 0
            assert res["diff"][("max", "best_single")]["ci"][0] > 0
        else:
            assert res["point"]["max"] < res["point"]["fit"] and hi < 0
        assert res["best_single_name"] in names


def test_peak_decomposition():
    rng = np.random.default_rng(1)
    names, beta = ["c0", "c1", "c2"], np.array([1.0, 1.0, -1.0])
    n_pass, per = 60, 8
    pidx = np.repeat(np.arange(n_pass), per)
    S = 0.3 * rng.normal(size=(len(pidx), 3))
    y = np.zeros(n_pass, bool); y[:30] = True
    category = np.array(["hardcode"] * 10 + ["harness"] * 10 + ["cancel"] * 10 + [None] * 30, object)
    for q in range(30):
        t = q * per + 3
        if category[q] == "hardcode":
            S[t, 0] += 5
        elif category[q] == "harness":
            S[t, 1] += 5
        else:  # large opposite-sign contributions that mostly cancel
            S[t, 1] += 10
            S[t, 2] += 8
    s_v = S @ beta + 0.3 * rng.normal(size=len(pidx))
    tok = {"v": s_v, **{n: S[:, i] for i, n in enumerate(names)}}

    recs = peak_decomposition(tok, "v", names, beta, pidx, y, category=category)
    assert len(recs) == 30
    for r in recs:
        assert np.isclose(r["contrib"].sum() + r["residual"], r["elevation"])
        assert r["token"] == r["passage"] * per + 3
        assert r["dominant_share"] <= r["top2_share"] <= 1
    summ = summarize_peaks(recs)
    assert summ["hardcode"]["dominant_counts"] == {"c0": 10}
    assert summ["harness"]["dominant_counts"] == {"c1": 10}
    assert summ["hardcode"]["residual_share"] < 0.2
    assert summ["cancel"]["n_opposite_sign"] == 10 and summ["hardcode"]["n_opposite_sign"] == 0
    assert summ["cancel"]["opposite_sign_clusters"] == {("c1", "c2"): 10}
    assert summ["all"]["n"] == 30
