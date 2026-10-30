# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import Callable

import pandas as pd

# to reduce effect of numeric errors in scores, before calculating rank features the scores
# are rounded with `round(RANK_FEATURES_PRECISION)`
# this does not eliminate the effect completely (since there could be some scores near the rounding border)
# but in most case this works fine
RANK_FEATURES_PRECISION = 5


def rank(df, c, uid_col):
    return group_by_uid_rounded(df, c, uid_col).rank(ascending=False, method="first")


def top2_dist(df, c, uid_col):
    groups = group_by_uid(df, c, uid_col)
    second = df[c].where(groups.rank(ascending=False, method="first") == 2)
    dist = groups.transform("max") - second.groupby(df[uid_col]).transform("max")
    # NaN if the group has fewer than 2 scores, except that a group of one row with a score has distance 0
    return dist.mask((groups.transform("size") == 1) & df[c].notna(), 0)


def dist_to_max(df, c, uid_col):
    return group_by_uid(df, c, uid_col).transform("max") - df[c]


def dist_to_min(df, c, uid_col):
    return df[c] - group_by_uid(df, c, uid_col).transform("min")


def feat_ptp(df, c, uid_col):
    groups = group_by_uid(df, c, uid_col)
    # max and min skip NaN, but a group with any NaN has no range
    return (groups.transform("max") - groups.transform("min")).where(groups.transform("count") == groups.transform("size"))


def diff_to_next(df, c, uid_col):
    return group_by_uid_rounded(df, c, uid_col).diff(1).abs()


def diff_to_prev(df, c, uid_col):
    return group_by_uid_rounded(df, c, uid_col).diff(-1)


def calc_rank_features(
    df: pd.DataFrame,
    funcs: dict[str, Callable],
    score_columns: list[str] | None,
    uid_col: str = "uid",
    fillna: int = -1,
) -> pd.DataFrame:
    if score_columns is None:
        score_columns = ["cossim_n3", "cossim_w"]
    res = pd.DataFrame(index=df.index)
    for c in score_columns:
        for column, func in funcs.items():
            res[f"{c}_{column}"] = func(df, c, uid_col).fillna(fillna).astype("int8" if "rank" in column else "float32")
    return res


def calc_diff_features(
    df: pd.DataFrame,
    funcs: dict[str, Callable],
    score_columns: list[str] | None,
    uid_col: str = "uid",
    fillna: int = -1,
) -> pd.DataFrame:
    if score_columns is None:
        score_columns = ["cossim_n3", "cossim_w"]
    res = pd.DataFrame(index=df.index)
    for c in score_columns:
        curr = df[[uid_col, c]].copy()
        curr[c] = curr[c].round(RANK_FEATURES_PRECISION)
        curr = curr.sort_values(by=[uid_col, c], ascending=[True, False])

        for column, func in funcs.items():
            res[f"{c}_{column}"] = func(curr, c, uid_col).fillna(fillna).astype("float32")
    return res


def group_by_uid(df, c, uid_col):
    # aggregates candidates using name to match UID (uid_col)
    return df.groupby(uid_col, group_keys=False)[c]


def group_by_uid_rounded(df, c, uid_col):
    return df[c].round(RANK_FEATURES_PRECISION).groupby(df[uid_col])
