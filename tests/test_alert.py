import pytest
from eepy_car.alert import DriverState, AlertLevel, AlertManager
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
        "yaw": 0.4,
        "pitch": 0.6
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
        base = dt.datetime.now()
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


class TestAlertManager:
    def test_init_sets_defaults(self):
        """Should return the correct defaults"""
        events = []
        manager = AlertManager(config, on_alert=events.append)

        assert manager.current_alert == AlertLevel.NONE
        assert manager.last_alert_time == dt.datetime.min
        assert manager.cooldown_seconds == pytest.approx(3.0)

    def test_scores_are_weighted_correctly(self):
        """Should return correct scores post weights"""
        manager = AlertManager(config, on_alert=print)
        state = DriverState(config)
        state.ear_score = 1.0
        state.mar_score = 2.0
        state.yaw_score = 10.0
        state.pitch_score = 5.0

        assert manager._drowsiness_score(state) == pytest.approx(1.3)
        assert manager._distraction_score(state) == pytest.approx(7.0)

    def test_evaluate_triggers_drowsiness_warning_and_callback(self):
        """Should correctly callback function with drowsiness warning"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)
        state.ear_score = 1.0
        state.mar_score = 0.0

        now = dt.datetime.now()
        level = manager.evaluate(state, now)

        assert level == AlertLevel.DROWSINESS_WARNING
        assert manager.current_alert == AlertLevel.DROWSINESS_WARNING
        assert manager.last_alert_time == now
        assert events == [AlertLevel.DROWSINESS_WARNING]

    def test_cooldown_blocks_alert_change(self):
        """Should prevent a second alert callback triggering within cooldown"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)

        base = dt.datetime.now()
        state.ear_score = 1.0
        manager.evaluate(state, base)

        state.ear_score = 2.0
        level = manager.evaluate(state, base + dt.timedelta(seconds=1))

        assert level == AlertLevel.DROWSINESS_WARNING
        assert events == [AlertLevel.DROWSINESS_WARNING]

    def test_after_cooldown_can_escalate_to_critical(self):
        """Should correctly alert when alert level increased"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)

        base = dt.datetime.now()
        state.ear_score = 1.0
        manager.evaluate(state, base)

        state.ear_score = 2.0
        level = manager.evaluate(state, base + dt.timedelta(seconds=4))

        assert level == AlertLevel.CRITICAL_DROWSINESS
        assert events == [
            AlertLevel.DROWSINESS_WARNING,
            AlertLevel.CRITICAL_DROWSINESS,
        ]

    def test_priority_prefers_critical_drowsiness_over_critical_distraction(self):
        """Should callback with drowsiness warning over distraction"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)
        state.ear_score = 2.0
        state.mar_score = 2.0
        state.yaw_score = 100.0
        state.pitch_score = 100.0

        level = manager.evaluate(state, dt.datetime.now())

        assert level == AlertLevel.CRITICAL_DROWSINESS
        assert events == [AlertLevel.CRITICAL_DROWSINESS]

    def test_exact_critical_threshold_does_not_trigger_critical(self):
        """Should not trigger alert when on limit"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)
        state.ear_score = 1.0
        state.mar_score = 1.0

        level = manager.evaluate(state, dt.datetime.now())

        assert level == AlertLevel.DROWSINESS_WARNING
        assert events == [AlertLevel.DROWSINESS_WARNING]

    def test_zero_scores_reset_alert_to_none_even_during_cooldown(self):
        """Should callback alert with None"""
        events = []
        manager = AlertManager(config, on_alert=events.append)
        state = DriverState(config)

        base = dt.datetime.now()
        state.ear_score = 1.0
        manager.evaluate(state, base)
        assert manager.current_alert == AlertLevel.DROWSINESS_WARNING

        state.ear_score = 0.0
        state.mar_score = 0.0
        state.yaw_score = 0.0
        state.pitch_score = 0.0
        level = manager.evaluate(state, base + dt.timedelta(seconds=1))

        assert level == AlertLevel.NONE
        assert manager.current_alert == AlertLevel.NONE
