"""Time to event metrics."""

import numpy as np

from octopus.types import MLType

from .config import Metric
from .core import Metrics


def _harrell_concordance_index(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Harrell's concordance index for risk scores.

    Wraps lifelines' concordance_index, handling the sign convention:
    - Our models output risk scores where higher = higher risk (shorter survival)
    - lifelines expects predicted_scores where higher = longer survival
    - We negate the estimate to bridge the convention

    Args:
        event_indicator: Boolean array, True = event observed.
        event_time: Float array, observed times.
        estimate: Risk scores from model (higher = higher risk).

    Returns:
        C-index in [0, 1]. 1.0 = perfect concordance, 0.5 = random.
    """
    try:
        from lifelines.utils import concordance_index  # noqa: PLC0415
    except ModuleNotFoundError as ex:
        from octopus.exceptions import OptionalImportError  # noqa: PLC0415

        raise OptionalImportError(
            "lifelines is required for concordance index metrics but is not installed. "
            'Install survival dependencies with: pip install "octopus-automl[survival]"'
        ) from ex

    result: float = concordance_index(
        event_times=event_time,
        predicted_scores=-estimate,
        event_observed=event_indicator,
    )
    return result


def _kaplan_meier_censoring_survival(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate censoring survival function G(t) via Kaplan-Meier.

    For the censoring distribution, the "events" are censorings
    (i.e., delta_censoring = NOT event_indicator). The KM estimator
    then gives G(t) = P(C > t).

    Args:
        event_indicator: Bool array, True = event observed.
        event_time: Float array, observed times.

    Returns:
        Tuple of (unique_times, g_values) where g_values[i] = G(unique_times[i]).
        G(t) is the probability of not being censored by time t.
    """
    # For censoring distribution, "events" are censorings
    censoring_indicator = ~event_indicator

    order = np.argsort(event_time, kind="mergesort")
    sorted_time = event_time[order]
    sorted_censoring = censoring_indicator[order]

    # Vectorised computation: O(n log n) instead of O(n²)
    unique_times, counts = np.unique(sorted_time, return_counts=True)

    n = len(sorted_time)
    # Cumulative count of individuals before each unique time
    cum_before = np.concatenate(([0], np.cumsum(counts[:-1])))
    # Number at risk at each unique time: those with time >= t
    at_risk = n - cum_before

    # Count censorings at each unique time via searchsorted + bincount
    indices = np.searchsorted(unique_times, sorted_time, side="left")
    n_censored_per_unique = np.bincount(
        indices,
        weights=sorted_censoring.astype(float),
        minlength=len(unique_times),
    )

    # Kaplan-Meier survival estimate for censoring: G(t) = prod(1 - d_c(t)/n(t))
    hazard = np.where(at_risk > 0, n_censored_per_unique / at_risk, 0.0)
    g_values = np.cumprod(1.0 - hazard, dtype=float)

    return unique_times, g_values


def _get_censoring_weight(t: float, unique_times: np.ndarray, g_values: np.ndarray) -> float:
    """Get G(t) for a specific time using step function lookup.

    Args:
        t: Time to evaluate.
        unique_times: Sorted unique times from KM estimator.
        g_values: Corresponding G values.

    Returns:
        G(t) value. Returns 1.0 if t is before all unique times.
    """
    idx = np.searchsorted(unique_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(g_values[idx])


class _FenwickTree:
    """Fenwick tree (Binary Indexed Tree) for O(log n) prefix sum queries.

    Used internally by Uno's C-index for efficient concordance counting.
    Supports point updates and prefix sum queries in O(log n) time.

    Args:
        size: Number of elements (0-indexed).
    """

    __slots__ = ("_n", "_tree")

    def __init__(self, size: int) -> None:
        self._n = size
        self._tree = np.zeros(size + 1, dtype=float)

    def update(self, idx: int, value: float = 1.0) -> None:
        """Add ``value`` at position ``idx`` (0-based).

        Args:
            idx: 0-based index to update.
            value: Value to add.
        """
        i = idx + 1
        n = self._n + 1
        while i < n:
            self._tree[i] += value
            i += i & -i

    def prefix_sum(self, idx: int) -> float:
        """Return sum of values in ``[0, idx]`` (0-based, inclusive).

        Args:
            idx: 0-based upper bound (inclusive).

        Returns:
            Sum of values at positions 0 through idx.
        """
        if idx < 0:
            return 0.0
        s = 0.0
        i = idx + 1
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def total(self) -> float:
        """Return sum of all values.

        Returns:
            Total sum across all positions.
        """
        return self.prefix_sum(self._n - 1) if self._n > 0 else 0.0


def _uno_concordance_index(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Uno's concordance index with IPCW (Inverse Probability of Censoring Weighting).

    Unlike Harrell's C-index, Uno's C accounts for censoring bias by weighting
    concordant/discordant pairs with the inverse probability of censoring.

    Uses a sort + Fenwick tree approach for O(n log n) complexity instead of
    the naive O(n²) pairwise comparison.

    Note: the censoring distribution G(t) is estimated from the evaluation set
    itself, not from the training set. This is a known limitation that
    introduces a small bias.

    Args:
        event_indicator: Boolean array, True = event observed.
        event_time: Float array, observed times.
        estimate: Risk scores from model (higher = higher risk).

    Returns:
        C-index in [0, 1]. 1.0 = perfect concordance, 0.5 = random.
        Returns 0.5 as fallback for degenerate cases (too few samples,
        no events, or no admissible pairs).
    """
    event = np.asarray(event_indicator, dtype=bool)
    time = np.asarray(event_time, dtype=float)
    risk = np.asarray(estimate, dtype=float)

    n = len(event)
    if n < 2:
        return 0.5

    # Estimate censoring distribution from evaluation data
    unique_times, g_values = _kaplan_meier_censoring_survival(event, time)

    # Truncation time: largest event time (not censoring time)
    event_times_only = time[event]
    if len(event_times_only) == 0:
        return 0.5
    tau = np.max(event_times_only)

    # Vectorized computation of censoring weights G(t) for all samples
    idx = np.searchsorted(unique_times, time, side="right") - 1
    g_all = np.where(idx < 0, 1.0, g_values[idx]).astype(float)

    # Map risk scores to integer ranks for Fenwick tree indexing
    unique_risk, risk_rank = np.unique(risk, return_inverse=True)
    m = len(unique_risk)

    # Build mapping from original index to sorted-by-time position
    time_order = np.argsort(time, kind="mergesort")

    # Group samples by unique time over the SORTED array so that
    # group boundaries (time_group_start) are in the same index space as time_order.
    sorted_time = time[time_order]
    unique_time_vals, time_group_counts = np.unique(sorted_time, return_counts=True)
    time_group_start = np.concatenate(([0], np.cumsum(time_group_counts[:-1])))

    # Fenwick tree over risk ranks
    bit = _FenwickTree(m)

    numerator = 0.0
    denominator = 0.0

    # Process time groups from LARGEST to SMALLEST.
    # The BIT contains only samples with time strictly greater than the current group.
    for g_idx in range(len(unique_time_vals) - 1, -1, -1):
        start = time_group_start[g_idx]
        count = time_group_counts[g_idx]

        # Indices (in original array) belonging to this time group
        group_indices = time_order[start : start + count]

        # First: query the BIT for each event sample in this group
        # (BIT contains only samples with strictly larger time)
        for orig_idx in group_indices:
            if not event[orig_idx]:
                continue
            if time[orig_idx] > tau:
                continue

            g_ti = g_all[orig_idx]
            if g_ti <= 0.0:
                continue

            weight = 1.0 / (g_ti * g_ti)

            # Total samples with time > time[i] already in BIT
            total_after = bit.total()
            if total_after <= 0.0:
                continue

            denominator += weight * total_after

            r = risk_rank[orig_idx]
            # Count of samples in BIT with risk < risk[i] (concordant pairs)
            concordant = bit.prefix_sum(r - 1)
            # Count with risk == risk[i] (tied pairs)
            tied = bit.prefix_sum(r) - concordant

            numerator += weight * (concordant + 0.5 * tied)

        # Then: insert ALL samples in this group into BIT
        # (so they are available as "j" for smaller-time groups)
        for orig_idx in group_indices:
            bit.update(risk_rank[orig_idx])

    if denominator == 0.0:
        return 0.5

    return numerator / denominator


@Metrics.register("CI")
def cindex_metric() -> Metric:
    """Harrell's concordance index metric configuration."""
    return Metric(
        name="CI",
        metric_function=_harrell_concordance_index,
        ml_types=[MLType.TIMETOEVENT],
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="concordance_index",
    )


@Metrics.register("CI_UNO")
def cindex_uno_metric() -> Metric:
    """Uno's concordance index metric configuration."""
    return Metric(
        name="CI_UNO",
        metric_function=_uno_concordance_index,
        ml_types=[MLType.TIMETOEVENT],
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="concordance_index_uno",
    )
