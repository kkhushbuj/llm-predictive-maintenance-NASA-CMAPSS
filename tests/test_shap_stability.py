from ml.shap_stability import AGREEMENT_THRESHOLD, CV_THRESHOLD, reduce_runs


def _run(unit, top_sensor, importance):
    return {
        "unit": unit,
        "top_sensors": [{"sensor": top_sensor, "importance": importance}],
        "all_sensor_importance": {top_sensor: importance, "sensor_1": 0.01},
    }


def test_reduce_runs_marks_stable_when_all_runs_agree_and_importance_is_flat():
    units = [1]
    runs = [[_run(1, "sensor_11", 0.5)] for _ in range(5)]

    [record] = reduce_runs(runs, units)

    assert record["mode_top_sensor"] == "sensor_11"
    assert record["top_sensor_agreement_rate"] == 1.0
    assert record["mode_sensor_importance_cv"] == 0.0
    assert record["stable"] is True


def test_reduce_runs_marks_unstable_when_top_sensor_disagrees_across_runs():
    units = [1]
    runs = [
        [_run(1, "sensor_11", 0.5)],
        [_run(1, "sensor_11", 0.5)],
        [_run(1, "sensor_9", 0.4)],
        [_run(1, "sensor_9", 0.4)],
        [_run(1, "sensor_9", 0.4)],
    ]

    [record] = reduce_runs(runs, units)

    # mode is sensor_9 (3/5 runs), which is below the agreement threshold
    assert record["mode_top_sensor"] == "sensor_9"
    assert record["top_sensor_agreement_rate"] == 3 / 5
    assert record["top_sensor_agreement_rate"] < AGREEMENT_THRESHOLD
    assert record["stable"] is False


def test_reduce_runs_marks_unstable_when_importance_is_volatile():
    units = [1]
    # Same top sensor every run (100% agreement) but importance swings wildly.
    runs = [
        [_run(1, "sensor_11", 0.1)],
        [_run(1, "sensor_11", 0.9)],
        [_run(1, "sensor_11", 0.1)],
        [_run(1, "sensor_11", 0.9)],
        [_run(1, "sensor_11", 0.5)],
    ]

    [record] = reduce_runs(runs, units)

    assert record["top_sensor_agreement_rate"] == 1.0
    assert record["mode_sensor_importance_cv"] > CV_THRESHOLD
    assert record["stable"] is False


def test_reduce_runs_handles_multiple_engines_independently():
    units = [1, 2]
    runs = [
        [_run(1, "sensor_11", 0.5), _run(2, "sensor_7", 0.3)],
        [_run(1, "sensor_11", 0.5), _run(2, "sensor_14", 0.2)],
    ]

    records = {r["unit"]: r for r in reduce_runs(runs, units)}

    assert records[1]["top_sensor_agreement_rate"] == 1.0
    assert records[2]["top_sensor_agreement_rate"] == 0.5
