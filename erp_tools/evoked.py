"""
evoked.py
=========

Computes condition-level evoked responses and grand averages across subjects.

evoked = epochs averaged per condition — the basic unit of an ERP.
grand average = evoked responses averaged further across subjects — a group-level result.
"""

from typing import Dict, List

import mne
import numpy as np


def compute_condition_evokeds(
    epochs: mne.Epochs,
) -> Dict[str, mne.Evoked]:
    """Average epochs per condition to produce evoked responses.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with condition labels (created by make_epochs).

    Returns
    -------
    evokeds : dict
        {condition_name: Evoked} dictionary.

    Examples
    --------
    >>> evokeds = compute_condition_evokeds(epochs_clean)
    >>> evokeds['tone_high'].plot()
    >>> evokeds['tone_low'].plot()
    """
    # Average across epochs for each condition key in epochs.event_id
    evokeds = {
        condition: epochs[condition].average()
        for condition in epochs.event_id.keys()
    }
    return evokeds


def grand_average(
    subject_evokeds: List[mne.Evoked],
    weight_by_nave: bool = True,
) -> mne.Evoked:
    """Compute a grand average from evoked responses across subjects.

    Parameters
    ----------
    subject_evokeds : list of mne.Evoked
        Evoked responses for the same condition from each subject.
        Example: [sub01 'tone_high', sub02 'tone_high', ...]
    weight_by_nave : bool
        If True, weight each evoked by its trial count (nave) before averaging.
        Recommended when trial counts vary across subjects.
        If False, use a simple unweighted average (each subject contributes equally).

    Returns
    -------
    grand_avg : mne.Evoked
        Grand-averaged evoked response.

    Notes
    -----
    - For statistical testing, pass per-subject evoked arrays directly to
      functions such as mne.stats.permutation_cluster_test rather than using
      the grand average. This function produces a representative waveform for
      visualisation purposes.
    - All subjects must have the same channel configuration. If channel sets
      differ across subjects, align them to a common subset beforehand.
    """
    weights = "nave" if weight_by_nave else "equal"
    grand_avg = mne.combine_evoked(subject_evokeds, weights=weights)
    return grand_avg


def compute_gfp(data_uv: np.ndarray) -> np.ndarray:
    """Global field power: population std (ddof=0) across channels at each time point.

    Parameters
    ----------
    data_uv : np.ndarray, shape (n_channels, n_times)

    Returns
    -------
    gfp : np.ndarray, shape (n_times,)
    """
    return data_uv.std(axis=0, ddof=0)


def compute_condition_gfp_grand_average(
    evokeds: Dict[str, mne.Evoked],
    picks: np.ndarray,
) -> np.ndarray:
    """Average GFP across conditions with each condition weighted equally.

    Each condition's evoked (already an average over its accepted trials) is
    converted to a GFP curve, then the curves are averaged with equal weight
    per condition — regardless of how many trials went into each condition's
    evoked. This avoids conditions with more accepted trials dominating the
    grand-average GFP, which a trial-level pool would not.

    Parameters
    ----------
    evokeds : dict of {condition_label: mne.Evoked}
        Condition-averaged evokeds to combine, e.g. one Evoked per
        intensity x frequency condition.
    picks : np.ndarray
        Channel indices to include in the GFP (see get_all_eeg_picks).

    Returns
    -------
    mean_gfp : np.ndarray, shape (n_times,)
        GFP curve averaged equally across the given conditions.
    """
    gfp_stack = [
        compute_gfp(ev.data[picks] * 1e6) for ev in evokeds.values()
    ]
    return np.mean(gfp_stack, axis=0)


def find_component_peak(
    times_ms: np.ndarray,
    gfp: np.ndarray,
    tmin_ms: float,
    tmax_ms: float,
) -> float:
    """Find a component's peak latency on a GFP curve within a search window.

    GFP (population std across channels) is non-negative by construction, so
    a component's polarity (+1/-1 in the COMPONENTS table) does not change how
    its peak is picked on the GFP curve: the peak is always the GFP maximum in
    the window, never the minimum. Polarity only matters when reading the peak
    amplitude off a signed, single-channel signal (e.g. peak_channel_amplitude_uv
    in notebook 03), not here.

    Parameters
    ----------
    times_ms : np.ndarray, shape (n_times,)
    gfp : np.ndarray, shape (n_times,)
    tmin_ms, tmax_ms : float
        Search window bounds, inclusive.

    Returns
    -------
    latency_ms : float
        Latency of the peak, or NaN if the window contains no samples.
    """
    mask = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)
    if not mask.any():
        return float('nan')
    peak_idx = int(np.where(mask)[0][np.argmax(gfp[mask])])
    return float(times_ms[peak_idx])
