import pytest
from eepy_car.alert import DriverState
import datetime as dt


config = {
    "thresholds": {
        "ear": {
            "value": 0.20,
            "decay_rate": 0.1
        },
        "mar": {
            "value": 0.50,
            "decay_rate": 0.15
        },
        "yaw_degrees": {
            "value": 25.0,
            "decay_rate": 5.0
        },
        "pitch_down_degrees": {
            "value": -20.0,
            "decay_rate": 5.0
        }
    },
    "weights": {
        "ear": 0.7,
        "mar": 0.3,
        "yaw": 0.6,
        "pitch": 0.4
    },
    "alert_limits": {
        "drowsiness_warning": 0.4,
        "drowsiness_critical": 1.0,
        "distraction_warning": 8.0,
        "distraction_critical": 20.0
    }
}


@pytest.fixture
def driver_state():
    return DriverState(config)


class TestDriverState:
    def test_init_sets_default_scores_and_thresholds(self):
        """Should set all scores to zero"""
        state = DriverState(config)
        assert state.ear_score == 0
        assert state.mar_score == 0
        assert state.yaw_score == 0
        assert state.pitch_score == 0
        assert state.thresholds == config["thresholds"]

    def test_accumulate_when_below_accumulates(self, driver_state):
        """Should correctly accumulate the score when the value is below threshold"""
        result = driver_state.accumulate_when_below(
            current_score=1.0,
            value=0.10,
            threshold_dict=driver_state.thresholds["ear"],
            t_delta=2.0,
        )
        assert result == pytest.approx(1.2)

    def test_accumulate_when_below_decays_and_clamps_to_zero(self, driver_state):
        """Should correctly decay the score down to 0 when value is above threshold"""
        result = driver_state.accumulate_when_below(
            current_score=0.1,
            value=0.30,
            threshold_dict=driver_state.thresholds["ear"],
            t_delta=2.0,
        )
        assert result == pytest.approx(0.0)

    def test_accumulate_when_above_accumulates(self, driver_state):
        """Should correctly accumulate when value is above threshold"""
        result = driver_state.accumulate_when_above(
            current_score=0.5,
            value=0.80,
            threshold_dict=driver_state.thresholds["mar"],
            t_delta=2.0,
        )
        assert result == pytest.approx(1.1)

    def test_accumulate_when_above_decays_and_clamps_to_zero(self, driver_state):
        """Should correctly decay to 0 when value below threshold"""
        result = driver_state.accumulate_when_above(
            current_score=0.2,
            value=0.40,
            threshold_dict=driver_state.thresholds["mar"],
            t_delta=2.0,
        )
        assert result == pytest.approx(0.0)

    def test_update_scores_accumulates_all_metrics(self, driver_state):
        """Should accumulate all scores"""
        base = dt.datetime.now()
        driver_state.last_t = base
        current_t = base + dt.timedelta(seconds=2)

        driver_state.update_scores(
            ear=0.10,
            mar=0.80,
            yaw=30.0,
            pitch=-30.0,
            current_t=current_t,
        )

        assert driver_state.ear_score == pytest.approx(0.2)
        assert driver_state.mar_score == pytest.approx(0.6)
        assert driver_state.yaw_score == pytest.approx(10.0)
        assert driver_state.pitch_score == pytest.approx(20.0)

    def test_update_scores_decays_and_clamps(self, driver_state):
        """Should decay all scores"""
        base = dt.datetime(2024, 1, 1, 0, 0, 0)
        driver_state.last_t = base
        driver_state.ear_score = 1.0
        driver_state.mar_score = 1.0
        driver_state.yaw_score = 1.0
        driver_state.pitch_score = 1.0

        driver_state.update_scores(
            ear=0.25,
            mar=0.40,
            yaw=10.0,
            pitch=-10.0,
            current_t=base + dt.timedelta(seconds=3),
        )

        assert driver_state.ear_score == pytest.approx(0.7)
        assert driver_state.mar_score == pytest.approx(0.55)
        assert driver_state.yaw_score == pytest.approx(0.0)
        assert driver_state.pitch_score == pytest.approx(0.0)

    def test_update_scores_uses_absolute_yaw(self, driver_state):
        """Should corretly correct yaw to positive value"""
        base = dt.datetime.now()
        driver_state.last_t = base

        driver_state.update_scores(
            ear=0.30,
            mar=0.40,
            yaw=-30.0,
            pitch=-10.0,
            current_t=base + dt.timedelta(seconds=1),
        )

        assert driver_state.yaw_score == pytest.approx(5.0)

    def test_update_scores_equal_to_threshold_decays_not_accumulates(self, driver_state):
        """Should decay scores at threshold"""
        base = dt.datetime.now()
        driver_state.last_t = base
        driver_state.ear_score = 1.0
        driver_state.mar_score = 1.0
        driver_state.yaw_score = 20.0
        driver_state.pitch_score = 20.0

        driver_state.update_scores(
            ear=0.20,
            mar=0.50,
            yaw=25.0,
            pitch=-20.0,
            current_t=base + dt.timedelta(seconds=2),
        )

        assert driver_state.ear_score == pytest.approx(0.8)
        assert driver_state.mar_score == pytest.approx(0.7)
        assert driver_state.yaw_score == pytest.approx(10.0)
        assert driver_state.pitch_score == pytest.approx(10.0)

    def test_update_scores_updates_last_t(self, driver_state):
        """Should update the last time with current time"""
        base = dt.datetime.now()
        current_t = base + dt.timedelta(seconds=1)
        driver_state.last_t = base

        driver_state.update_scores(
            ear=0.2,
            mar=0.5,
            yaw=25.0,
            pitch=-20.0,
            current_t=current_t,
        )

        assert driver_state.last_t == current_t

    def test_update_scores_zero_time_delta_keeps_scores_unchanged(self, driver_state):
        """Should have no change in score if no time elapsed"""
        base = dt.datetime.now()
        driver_state.last_t = base
        driver_state.ear_score = 1.0
        driver_state.mar_score = 1.0
        driver_state.yaw_score = 1.0
        driver_state.pitch_score = 1.0

        driver_state.update_scores(
            ear=0.0,
            mar=1.0,
            yaw=100.0,
            pitch=-100.0,
            current_t=base,
        )

        assert driver_state.ear_score == pytest.approx(1.0)
        assert driver_state.mar_score == pytest.approx(1.0)
        assert driver_state.yaw_score == pytest.approx(1.0)
        assert driver_state.pitch_score == pytest.approx(1.0)
