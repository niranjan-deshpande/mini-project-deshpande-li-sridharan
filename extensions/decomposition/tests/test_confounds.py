import numpy as np

from rhdecomp import auroc, confound_crosstab, confound_response_auroc, stratified_auroc


def test_confound_checks():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    c = np.array([1, 1, 0, 1, 0, 0, 0, 1, np.nan, np.nan])
    s = np.array([5, 4, 3, 4.5, 1, 2, 0, 3.5, 9, 9])
    ct = confound_crosstab(y, c)
    assert ct["n"] == {0: {0: 3, 1: 2}, 1: {0: 1, 1: 2}}
    assert np.isclose(ct["p_confound_given_hack"], 2 / 3) and np.isclose(ct["p_confound_given_clean"], 2 / 5)

    st = stratified_auroc(y, s, c)
    assert st[1]["n"] == 4 and st[1]["n_hack"] == 2 and st[0]["n_clean"] == 3
    assert np.isclose(st[1]["auroc"], auroc([1, 1, 0, 0], [5, 4, 4.5, 3.5]))
    assert st[0]["auroc"] == 1.0

    cr = confound_response_auroc(y, s, c)
    assert cr["n"] == 5 and cr["n_confound"] == 2
    assert cr["auroc"] == 1.0  # the clean confound passages score highest
