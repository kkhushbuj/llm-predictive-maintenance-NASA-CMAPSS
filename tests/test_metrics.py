import pytest

from ml.metrics import nasa_score, rmse


def test_rmse_zero_for_perfect_prediction():
    assert rmse([10, 20, 30], [10, 20, 30]) == 0.0


def test_rmse_matches_hand_computation():
    # errors [1, -2] -> mean squared error 2.5 -> sqrt ~= 1.581
    assert rmse([10, 20], [11, 18]) == pytest.approx(1.5811388300841898)


def test_nasa_score_zero_for_perfect_prediction():
    assert nasa_score([10, 20, 30], [10, 20, 30]) == 0.0


def test_nasa_score_penalizes_late_predictions_more_than_early():
    # Late prediction: predicted RUL too high (d > 0, exp(d/10)).
    late = nasa_score([50], [60])
    # Early prediction: predicted RUL too low by the same margin (d < 0, exp(-d/13)).
    early = nasa_score([50], [40])
    assert late > early
