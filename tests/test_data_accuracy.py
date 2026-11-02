import pytest

import data_accuracy_check as dac


@pytest.fixture(scope="module")
def results():
    return dac.run_all()


@pytest.mark.parametrize("subset", dac.SUBSETS)
def test_data_accuracy(results, subset):
    failures = [f"{name}: {detail}" for name, ok, detail in results[subset] if not ok]
    assert not failures, "\n".join(failures)
