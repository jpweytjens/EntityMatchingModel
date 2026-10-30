"""Pin the vectorised rank and diff features to the per-group code they replaced."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from emm.features.features_rank import (
    RANK_FEATURES_PRECISION,
    calc_diff_features,
    calc_rank_features,
    diff_to_next,
    diff_to_prev,
    dist_to_max,
    dist_to_min,
    feat_ptp,
    rank,
    top2_dist,
)
from emm.helper.io import load_pickle
from emm.supervised_model.pandas_supervised_model import PandasSupervisedLayerTransformer


def reference_group_by_uid(df, c, uid_col):
    return df.groupby(uid_col, group_keys=False)[c]


def reference_ptp(a):
    if a is None or len(a) == 0 or np.min(a) is None:
        return 0
    return np.ptp(a)


def reference_rank(df, c, uid_col):
    return reference_group_by_uid(df, c, uid_col).apply(
        lambda x: x.round(RANK_FEATURES_PRECISION).rank(ascending=False, method="first")
    )


def reference_top2_dist(df, c, uid_col):
    return reference_group_by_uid(df, c, uid_col).transform(lambda x: reference_ptp(x.nlargest(2)))


def reference_feat_ptp(df, c, uid_col):
    return reference_group_by_uid(df, c, uid_col).transform(reference_ptp)


def reference_diff_to_next(df, c, uid_col):
    return reference_group_by_uid(df, c, uid_col).apply(lambda x: x.round(RANK_FEATURES_PRECISION).diff(1).abs())


def reference_diff_to_prev(df, c, uid_col):
    return reference_group_by_uid(df, c, uid_col).apply(lambda x: x.round(RANK_FEATURES_PRECISION).diff(-1))


REFERENCE_RANK_FEATURES: dict[str, Callable] = {
    "rank": reference_rank,
    "top2_dist": reference_top2_dist,
    "dist_to_max": dist_to_max,
    "dist_to_min": dist_to_min,
    "ptp": reference_feat_ptp,
}
RANK_FEATURES: dict[str, Callable] = {
    "rank": rank,
    "top2_dist": top2_dist,
    "dist_to_max": dist_to_max,
    "dist_to_min": dist_to_min,
    "ptp": feat_ptp,
}
REFERENCE_DIFF_FEATURES: dict[str, Callable] = {"diff_to_next": reference_diff_to_next, "diff_to_prev": reference_diff_to_prev}
DIFF_FEATURES: dict[str, Callable] = {"diff_to_next": diff_to_next, "diff_to_prev": diff_to_prev}

nan = np.nan
GROUPS = {
    "nan_between_values": [0.5, nan, 0.2],
    "value_then_nan": [0.5, nan],
    "nan_then_value": [nan, 0.5],
    "one_value_two_nans": [0.5, nan, nan],
    "all_nan": [nan, nan, nan],
    "single_nan": [nan],
    "single_value": [0.7],
    "tie": [0.7, 0.7],
    "tie_on_top": [0.7, 0.2, 0.7],
    "two_values": [0.3, 0.100001],
    "rounding_border": [0.123455, 0.123445, 0.1234549, 0.1234551],
    "rounds_to_tie": [0.1000001, 0.1000002],
    "zeros": [0.0, 0.0, 0.0],
}


def frame_from(groups: list[list[float]], dtype: str) -> pd.DataFrame:
    uid = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    score = np.array([s for g in groups for s in g], dtype=dtype)
    return pd.DataFrame({"uid": uid, "score": score})


def assert_rank_and_diff_features_identical(df: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        calc_rank_features(df, funcs=RANK_FEATURES, score_columns=["score"]),
        calc_rank_features(df, funcs=REFERENCE_RANK_FEATURES, score_columns=["score"]),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        calc_diff_features(df, funcs=DIFF_FEATURES, score_columns=["score"]),
        calc_diff_features(df, funcs=REFERENCE_DIFF_FEATURES, score_columns=["score"]),
        check_exact=True,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("group", GROUPS.values(), ids=GROUPS.keys())
def test_features_identical_per_group(group, dtype):
    assert_rank_and_diff_features_identical(frame_from([group], dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_features_identical_across_groups(dtype):
    assert_rank_and_diff_features_identical(frame_from(list(GROUPS.values()), dtype))


@pytest.mark.parametrize("function", [rank, top2_dist, feat_ptp])
def test_functions_keep_dtype_and_index(function):
    df = frame_from(list(GROUPS.values()), "float32").sample(frac=1, random_state=0)
    reference = {rank: reference_rank, top2_dist: reference_top2_dist, feat_ptp: reference_feat_ptp}[function]
    pd.testing.assert_series_equal(function(df, "score", "uid"), reference(df, "score", "uid"), check_exact=True)


def test_functions_align_by_position_with_duplicate_index():
    # the candidate frame has a unique index, but the functions must not depend on it
    df = frame_from(list(GROUPS.values()), "float32")
    duplicated = df.set_axis(np.zeros(len(df), dtype=int))
    for function in [rank, top2_dist, feat_ptp, diff_to_next, diff_to_prev]:
        res = function(duplicated, "score", "uid")
        assert res.index.equals(duplicated.index)
        np.testing.assert_array_equal(res.to_numpy(), function(df, "score", "uid").to_numpy())


def test_features_on_empty_input():
    df = frame_from([], "float32")
    assert calc_rank_features(df, funcs=RANK_FEATURES, score_columns=["score"]).empty
    assert calc_diff_features(df, funcs=DIFF_FEATURES, score_columns=["score"]).empty


def test_top2_dist_needs_two_scores_unless_group_has_one_row():
    df = frame_from([[0.5, nan], [0.5, nan, 0.2], [0.5]], "float32")
    res = top2_dist(df, "score", "uid")
    assert res.iloc[:2].isna().all()
    assert (res.iloc[2:5] == np.float32(0.5) - np.float32(0.2)).all()
    assert res.iloc[5] == 0


def test_feat_ptp_propagates_nan():
    res = feat_ptp(frame_from([[0.5, nan, 0.2]], "float32"), "score", "uid")
    assert res.isna().all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("seed", range(20))
def test_features_identical_on_random_groups(seed, dtype):
    rng = np.random.default_rng(seed)
    n_rows = rng.integers(1, 400)
    df = pd.DataFrame(
        {
            "uid": rng.integers(0, max(n_rows // 5, 1), size=n_rows),
            # few distinct values, so that ties and rounding borders occur
            "score": rng.choice(rng.random(max(n_rows // 2, 1)).round(6), size=n_rows).astype(dtype),
        }
    )
    df.loc[rng.random(n_rows) < rng.choice([0.0, 0.2, 0.8]), "score"] = nan
    assert_rank_and_diff_features_identical(df)


def test_nm_score_identical_on_fitted_model(supervised_model):
    candidates = frame_from(list(GROUPS.values()), "float64").rename(columns={"score": "score_0"})
    candidates["gt_uid"] = np.arange(len(candidates))
    for prefix in ["", "gt_"]:
        candidates[f"{prefix}entity_id"] = np.arange(len(candidates))
    candidates["name"] = candidates["preprocessed"] = "rare foo company"
    candidates["gt_name"] = candidates["gt_preprocessed"] = "rare bar company ltd"

    def score(rank_features=None, diff_features=None):
        sm = load_pickle(supervised_model[2].name, supervised_model[2].parent)
        feat = sm.named_steps["feat"]
        # the pickle holds the functions by reference, so it resolves to the current code
        assert feat.rank_features["top2_dist"] is top2_dist
        if rank_features is not None:
            feat.rank_features, feat.diff_features, feat.funcs = rank_features, diff_features, None
        tr = PandasSupervisedLayerTransformer({"nm_score": {"model": sm, "enable": True}}, return_features=True)
        return tr.transform(candidates.copy())

    pd.testing.assert_frame_equal(score(), score(REFERENCE_RANK_FEATURES, REFERENCE_DIFF_FEATURES), check_exact=True)
