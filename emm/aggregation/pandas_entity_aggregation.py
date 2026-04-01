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

from functools import partial
from typing import Literal

import pandas as pd
from sklearn.base import TransformerMixin

from emm.aggregation.base_entity_aggregation import BaseEntityAggregation, matching_max_candidate
from emm.loggers import Timer


class PandasEntityAggregation(TransformerMixin, BaseEntityAggregation):
    """Pandas name-matching aggregation code"""

    def __init__(
        self,
        score_col: str,
        account_col: str = "account",
        index_col: str = "entity_id",
        gt_entity_id_col: str = "gt_entity_id",
        uid_col: str = "uid",
        gt_uid_col: str = "gt_uid",
        name_col: str = "name",
        freq_col: str = "counterparty_account_count_distinct",
        output_col: str = "agg_score",
        preprocessed_col: str = "preprocessed",
        gt_name_col: str = "gt_name",
        gt_preprocessed_col: str = "gt_preprocessed",
        correct_col: str = "correct",
        aggregation_method: Literal["max_frequency_nm_score", "mean_score"] = "max_frequency_nm_score",
        blacklist: list[str] | None = None,
    ) -> None:
        """Pandas name-matching aggregation code

        Last and optional step in PandasEntityMatching.

        Optionally, the EMM package can also be used to match a group of company names that belong together,
        to a company name in the ground truth. (For example, all names used to address an external bank account.)

        This step makes use of name-matching scores from the supervised layer. We refer to this as the aggregation step.
        (This step is not needed for standalone name matching.)

        The `account_col` column indicates which names-to-match belong together.
        The combination of scores is based on `score_col`, e.g. the name-matching score `nm_score`.

        Two aggregation methods are available:

        - "mean_score": takes the mean score from all names-to-match to find the best ground-truth name.
        - "max_frequency_nm_score": weights the nm_score with the frequency and takes the maximum to find the best
            ground-truth name.

        Args:
            score_col: name-matching score "nm_score" or first cosine similarity score "score_0".
            account_col: account column, default is "account".
            index_col: id column, default is "entity_id".
            gt_entity_id_col: ground truth id column, default is "gt_entity_id".
            uid_col: uid column, default is "uid".
            gt_uid_col: ground truth uid column, default is "gt_uid".
            name_col: name column, default is "name".
            freq_col: name frequency column, default is "counterparty_account_count_distinct".
            output_col: Name of column to store the final score
            preprocessed_col: Name of column of preprocessed input
            gt_name_col: ground truth name column, default is "gt_name".
            gt_preprocessed_col: column name of preprocessed ground truth names, default is "preprocessed".
            correct_col: column indicating correct matches, if present. default is "correct". optional.
            aggregation_method: default is "max_frequency_nm_score", alternative is "mean_score".
            blacklist: blacklist of names to skip in clustering.
        """
        BaseEntityAggregation.__init__(
            self,
            score_col=score_col,
            account_col=account_col,
            index_col=index_col,
            gt_entity_id_col=gt_entity_id_col,
            uid_col=uid_col,
            gt_uid_col=gt_uid_col,
            name_col=name_col,
            freq_col=freq_col,
            output_col=output_col,
            preprocessed_col=preprocessed_col,
            gt_name_col=gt_name_col,
            gt_preprocessed_col=gt_preprocessed_col,
            correct_col=correct_col,
            aggregation_method=aggregation_method,
            blacklist=blacklist or [],
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TransformerMixin:
        """Dummy function, no fitting is required."""
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Only calls transform(), no fitting required"""
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Vectorized version of transform().

        Replaces the per-account groupby().apply(matching_max_candidate) with
        bulk vectorized operations for both mean_score and max_frequency_nm_score.
        """
        if X is None:
            return None

        with Timer("PandasEntityAggregation.transform") as timer:
            timer.log_param("n", len(X))

            group = self.get_group(X)

            # Separate rows with matches from those without
            has_match = ~X[self.gt_uid_col].isna()

            # =====================================================================
            # Handle single-candidate accounts (no aggregation needed)
            # =====================================================================
            grouped = X[has_match].groupby(group)
            single_mask = grouped[self.gt_uid_col].transform("count") == 1
            one_match_df = X[has_match][single_mask].copy()
            one_match_df[self.output_col] = one_match_df[self.score_col]
            one_match_df["freq_score"] = (
                one_match_df[self.score_col] * one_match_df[self.freq_col]
            )

            # =====================================================================
            # Handle multi-candidate accounts (need aggregation)
            # =====================================================================
            mpl_match_df = X[has_match][~single_mask].copy()

            # Filter blacklisted names
            mpl_match_df = self.remove_blacklisted_names(
                df=mpl_match_df, preprocessed_col=self.preprocessed_col
            )

            if len(mpl_match_df) > 0:
                gt_group = self.get_gt_group()

                if self.aggregation_method == "mean_score":
                    cl_match_df = self._vectorized_mean_score(mpl_match_df, group, gt_group)
                else:
                    cl_match_df = self._vectorized_max_freq(mpl_match_df, group, gt_group)
            else:
                cl_match_df = mpl_match_df

            # =====================================================================
            # Combine results
            # =====================================================================
            res = pd.concat([one_match_df, cl_match_df])

            assert self.output_col in res.columns
            res["best_match"] = True
            res["best_rank"] = 1
            timer.log_param("cands", len(res))

        return res

    def _vectorized_mean_score(self, df, group, gt_group):
        """Vectorized mean_score: average score per gt match, pick best per account."""
        # Mean score per gt match, broadcast back to each row
        df[self.output_col] = df.groupby(gt_group, dropna=False)[self.score_col].transform("mean")
        # Pick best gt match per account: highest agg_score, break ties by raw score
        df = df.sort_values([self.output_col, self.score_col], ascending=False)
        return df.drop_duplicates(subset=group, keep="first")

    def _vectorized_max_freq(self, df, group, gt_group):
        """Vectorized max_frequency_nm_score: frequency-weighted score, pick best per account."""
        # Step 1: freq-weighted score per row
        df["freq_score"] = df[self.freq_col] * df[self.score_col]

        # Step 2: Aggregate by gt_group (e.g. [gt_entity_id, gt_uid, account])
        agg_df = (
            df.groupby(gt_group, dropna=False)
            .agg({self.freq_col: "sum", "freq_score": "sum"})
            .reset_index()
        )
        agg_df[self.output_col] = agg_df["freq_score"] / agg_df[self.freq_col]

        # Step 3: Best gt match per account (highest freq_score)
        account_cols = [self.account_col]
        idx_best = agg_df.groupby(account_cols)["freq_score"].idxmax()
        best_per_account = agg_df.loc[idx_best]

        # Step 4: Join back to get original row data
        cl_match_df = df.merge(
            best_per_account[gt_group + [self.output_col]],
            on=gt_group,
            how="inner",
        )

        # Pick one representative row per account (highest freq_score)
        cl_match_df = cl_match_df.sort_values("freq_score", ascending=False)
        return cl_match_df.drop_duplicates(subset=account_cols, keep="first")

    def remove_blacklisted_names(self, df: pd.DataFrame, preprocessed_col: str = "preprocessed"):
        # filter out all processed names that are in blacklist or empty.
        # idea: these are too generic/not-good to use for account matching anyway.
        if preprocessed_col in df.columns:
            # preprocessed column should always be present
            return df.loc[~df[preprocessed_col].isin([*self.blacklist, ""])]
        return df
