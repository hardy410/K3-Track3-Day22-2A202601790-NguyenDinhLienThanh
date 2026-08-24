import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_manual_value() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    expected_logit = 0.1 * ((-0.5 + 1.5) - (-0.6 + 1.0))
    assert loss == pytest.approx(np.logaddexp(0.0, -expected_logit))


def test_dpo_loss_is_stable_for_extreme_margins() -> None:
    loss = dpo_loss(
        np.array([-1.0, -10_000.0]),
        np.array([-10_000.0, -1.0]),
        np.array([-1.0, -1.0]),
        np.array([-1.0, -1.0]),
        beta=1.0,
    )
    assert np.isfinite(loss)


def test_dpo_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dpo_loss(
            np.array([-0.5]),
            np.array([-1.0, -2.0]),
            np.array([-0.5]),
            np.array([-1.0]),
            beta=0.1,
        )


def test_orpo_loss_combines_sft_and_preference_penalty() -> None:
    sft_nll = np.array([1.0, 2.0])
    loss = orpo_loss(
        sft_nll,
        np.array([-0.5, -0.4]),
        np.array([-1.5, -1.2]),
        lambda_orpo=0.1,
    )
    assert np.isfinite(loss)
    assert loss > np.mean(sft_nll)


def test_orpo_zero_weight_equals_mean_sft_loss() -> None:
    loss = orpo_loss(
        np.array([1.0, 3.0]),
        np.array([-0.5, -0.4]),
        np.array([-1.5, -1.2]),
        lambda_orpo=0.0,
    )
    assert loss == pytest.approx(2.0)
