from __future__ import annotations

import numpy as np


def _validated_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(array, dtype=np.float64) for array in arrays)
    if not converted or converted[0].size == 0:
        raise ValueError("loss inputs must not be empty")
    expected_shape = converted[0].shape
    if any(array.shape != expected_shape for array in converted[1:]):
        raise ValueError("all loss inputs must have the same shape")
    if any(not np.all(np.isfinite(array)) for array in converted):
        raise ValueError("loss inputs must contain only finite values")
    return converted


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    Uses ``logaddexp(0, -x)`` as a numerically stable ``-log(sigmoid(x))``.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite value")
    policy_chosen, policy_rejected, ref_chosen, ref_rejected = _validated_arrays(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return float(np.mean(np.logaddexp(0.0, -logits)))


def _log_odds_from_log_probability(log_probability: np.ndarray) -> np.ndarray:
    # A sequence log-probability of exactly zero implies infinite odds. Clamp by
    # machine epsilon to retain a finite, well-defined training objective.
    clipped = np.minimum(log_probability, -np.finfo(np.float64).eps)
    threshold = -np.log(2.0)
    lower_branch = clipped < threshold
    log_one_minus_probability = np.empty_like(clipped)
    log_one_minus_probability[lower_branch] = np.log1p(-np.exp(clipped[lower_branch]))
    log_one_minus_probability[~lower_branch] = np.log(-np.expm1(clipped[~lower_branch]))
    result: np.ndarray = clipped - log_one_minus_probability
    return result


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    The objective is mean SFT NLL plus a weighted logistic penalty on the
    chosen-vs-rejected log-odds ratio.
    """
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0:
        raise ValueError("lambda_orpo must be a non-negative finite value")
    nll, chosen, rejected = _validated_arrays(sft_nll, chosen_logps, rejected_logps)
    if np.any(nll < 0):
        raise ValueError("sft_nll must be non-negative")
    if np.any(chosen > 0) or np.any(rejected > 0):
        raise ValueError("log probabilities must be less than or equal to zero")

    log_odds_ratio = _log_odds_from_log_probability(chosen) - _log_odds_from_log_probability(
        rejected
    )
    preference_penalty = np.logaddexp(0.0, -log_odds_ratio)
    return float(np.mean(nll) + lambda_orpo * np.mean(preference_penalty))
