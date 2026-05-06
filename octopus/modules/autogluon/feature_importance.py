"""OOF permutation feature importance for AutoGluon.

The AG module produces FI on the DEV partition only; computing FI on test
data is `TaskPredictor`'s responsibility, not the module's.

Algorithm
---------

1. Enumerate level-1 children via `predictor.model_names(level=1)`.
2. Keep only children for which `_bagged_mode is True and not _child_oof`.
   These are the children that expose OOF predictions usable for permutation
   FI without a held-out dataset (random forests / extra trees do not
   qualify because AG fits them in non-bagged mode with `_child_oof=True`).
3. Pick the leaderboard-best survivor.
4. Call `predictor.feature_importance(data=None, model=best, feature_stage="transformed_model", ...)`.
   `feature_stage="original"` is forbidden with `data=None`; `feature_stage="transformed"`
   raises for bagged models.
5. Aggregate transformed-feature importances back to original feature names
   via `predictor._learner.feature_generator.get_feature_links()`. Permutation
   importance is additive in expectation, so summing transformed children per
   original is the conservative-honest reduction.

If no qualifying L1 child exists (e.g. AG only fitted RF/XT models), return
an empty DataFrame and log a warning - never fall back to test-data FI, and
never fail the whole module on optional FI.

AG private API surface
----------------------

This module reads two AG-internal attributes that have no public equivalent
in current AG releases:

* `predictor._trainer.load_model(name)` to inspect a child's `_bagged_mode`
  and `_child_oof` flags.
* `predictor._learner.feature_generator.get_feature_links()` to map
  transformed columns back to originals.

If either breaks across an AG upgrade, `compute_fi` will raise (caught by
`Module.fit` callers as a hard FI failure rather than silent corruption).

Reproducibility
---------------

AG's permutation FI uses `np.random` internally and exposes no `seed=` parameter.
This function therefore does not attempt to seed RNG; the resulting FI ranking
is non-deterministic across runs (typically stable for top features, noisy in
the tail). If reproducibility matters for a downstream consumer, raise
`_FI_NUM_SHUFFLE_SETS` (module constant) to tighten the confidence band.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from octopus._optional.autogluon import TabularPredictor
from octopus.logger import get_logger

logger = get_logger()

_FI_SUBSAMPLE_SIZE: Final[int] = 5000
_FI_NUM_SHUFFLE_SETS: Final[int | None] = None
_FI_CONFIDENCE_LEVEL: Final[float] = 0.99


def compute_fi(
    predictor: TabularPredictor,
    *,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
    leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Compute OOF permutation feature importance, aggregated to original feature names.

    Args:
        predictor: Fitted AutoGluon predictor.
        feature_cols: Original feature column names. The published FI frame's
            feature names are guaranteed to be a subset of this list.
        feature_groups: Caller-supplied feature groups. AG cannot compute group-level
            FI; if non-empty, a single warning is logged.
        leaderboard: Pre-fetched leaderboard from `predictor.leaderboard(silent=True)`,
            used to pick the best qualifying L1 child by validation score.

    Returns:
        DataFrame indexed by original feature name with a single `importance` column,
        sorted by importance descending. The result also carries
        `df.attrs["training_id"] = "<actual_l1_model_name>"`. Returns an empty
        DataFrame when no qualifying L1 child is available.
    """
    if feature_groups:
        logger.warning(
            "Group feature importances are not supported by AutoGluon; computing per-feature importances only."
        )

    best_l1 = _select_oof_l1_child(predictor, leaderboard)
    if best_l1 is None:
        return pd.DataFrame()

    logger.info("Computing OOF permutation FI via L1 model: %s", best_l1)
    fi_transformed: pd.DataFrame = predictor.feature_importance(
        data=None,
        model=best_l1,
        feature_stage="transformed_model",
        subsample_size=_FI_SUBSAMPLE_SIZE,
        time_limit=None,
        include_confidence_band=True,
        confidence_level=_FI_CONFIDENCE_LEVEL,
        num_shuffle_sets=_FI_NUM_SHUFFLE_SETS,
        silent=True,
    )

    fi_df = _aggregate_transformed_to_original(
        fi_transformed,
        predictor=predictor,
        feature_cols=feature_cols,
        training_id=best_l1,
    )
    return fi_df.sort_values(by="importance", ascending=False)


def _select_oof_l1_child(predictor: TabularPredictor, leaderboard: pd.DataFrame) -> str | None:
    """Return the leaderboard-best L1 child supporting OOF FI, or None."""
    l1_models = predictor.model_names(level=1)
    if not l1_models:
        logger.info("No level-1 models found; cannot compute OOF FI.")
        return None

    qualifying: list[str] = []
    for name in l1_models:
        try:
            child = predictor._trainer.load_model(name)
        except Exception as exc:
            logger.debug("Skipping L1 child %s (load failed: %s)", name, exc)
            continue
        if getattr(child, "_bagged_mode", False) and not getattr(child, "_child_oof", False):
            qualifying.append(name)

    if not qualifying:
        logger.warning(
            "No qualifying L1 child for OOF FI (need _bagged_mode=True and not _child_oof). Returning empty FI."
        )
        return None

    l1_leaderboard = leaderboard[leaderboard["model"].isin(qualifying)]
    if l1_leaderboard.empty:
        logger.warning("No qualifying L1 child appears in leaderboard; returning empty FI.")
        return None
    best_l1: str = l1_leaderboard.iloc[0]["model"]
    return best_l1


def _aggregate_transformed_to_original(
    fi_transformed: pd.DataFrame,
    *,
    predictor: TabularPredictor,
    feature_cols: list[str],
    training_id: str,
) -> pd.DataFrame:
    """Sum transformed-feature importances back into the original feature space."""
    links: dict[str, list[str]] = predictor._learner.feature_generator.get_feature_links()
    transformed_to_original = {t: o for o, ts in links.items() for t in ts}

    unmapped = set(fi_transformed.index) - set(transformed_to_original) - set(feature_cols)
    if unmapped:
        raise RuntimeError(
            f"AutoGluon returned permutation FI for transformed features {sorted(unmapped)} "
            f"that are absent from feature_generator.get_feature_links() and from feature_cols. "
            f"This indicates a get_feature_links contract change in AutoGluon "
            f"({predictor.__class__.__module__}); aggregating would silently drop these importances."
        )

    agg = (
        fi_transformed["importance"]
        .rename(index=transformed_to_original)
        .groupby(level=0)
        .sum()
        .reindex(feature_cols, fill_value=0.0)
        .to_frame()
    )
    agg.index.name = "feature"
    agg.attrs["training_id"] = training_id
    return agg
