import pandas as pd

from ml.preprocessing import add_rul_test, add_rul_train


def test_add_rul_train_counts_down_to_zero_at_last_cycle():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    out = add_rul_train(df)
    assert out.loc[out["unit"] == 1, "RUL"].tolist() == [2, 1, 0]
    assert out.loc[out["unit"] == 2, "RUL"].tolist() == [1, 0]


def test_add_rul_train_caps_rul_at_configured_max():
    from ml.config import RUL_CAP

    df = pd.DataFrame({"unit": [1] * 5, "cycle": list(range(1, 6))})
    out = add_rul_train(df)
    assert out["RUL"].max() <= RUL_CAP


def test_add_rul_test_back_computes_from_given_final_rul():
    test = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 1]})
    rul_test = pd.DataFrame({"RUL": [10, 20]})  # unit 1 -> 10, unit 2 -> 20

    out = add_rul_test(test, rul_test)

    # unit 1: truncated at cycle 2, 1 cycle before that -> RUL = 10 + 1 = 11 at cycle 1
    assert out.loc[(out["unit"] == 1) & (out["cycle"] == 2), "RUL"].item() == 10
    assert out.loc[(out["unit"] == 1) & (out["cycle"] == 1), "RUL"].item() == 11
    assert out.loc[(out["unit"] == 2) & (out["cycle"] == 1), "RUL"].item() == 20
