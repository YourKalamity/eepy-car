import datetime as dt
from enum import Enum
from typing import Callable
from eepy_car.alert import DriverState


class AlertLevel(Enum):
    NONE = 0
    DISTRACTION_WARNING = 1
    DROWSINESS_WARNING = 2
    CRITICAL_DISTRACTION = 3
    CRITICAL_DROWSINESS = 4


class AlertManager:
    """
    Evaluates DriverState scores using weighted values to calculate drowsiness and distraction scores
    Alerts when values exceed threshold
    """

    def __init__(self, config: dict, on_alert: Callable[[AlertLevel], None]) -> None:
        """Initialises the AlertManager with weights and limits from config.

        Args:
            config (dict): The configuration dictionary loaded from file.
        """
        self.weights = config["weights"]
        self.limits = config["alert_limits"]
        self.current_alert = AlertLevel.NONE
        self.last_alert_time = dt.datetime.min
        self.cooldown_seconds = config.get("cooldown_seconds", 3.0)
        self.on_alert = on_alert

    def _drowsiness_score(self, state: DriverState) -> float:
        """Computes the weighted composite drowsiness score.

        Args:
            state (DriverState): The current driver state.

        Returns:
            float: Weighted sum of EAR and MAR scores.
        """
        return (state.ear_score * self.weights["ear"] + state.mar_score * self.weights["mar"])

    def _distraction_score(self, state: DriverState) -> float:
        """Computes the weighted composite distraction score.

        Args:
            state (DriverState): The current driver state.

        Returns:
            float: Weighted sum of yaw and pitch scores.
        """
        return (state.yaw_score * self.weights["yaw"] + state.pitch_score * self.weights["pitch"])

    def evaluate(self, state: DriverState, now: dt.datetime) -> AlertLevel:
        """Evaluates the current driver state and returns the alert level.

        Args:
            state (DriverState): The current driver state with accumulated scores.
            now (dt.datetime): The current timestamp.

        Returns:
            AlertLevel: The current alert level.
        """
        drowsiness = self._drowsiness_score(state)
        distraction = self._distraction_score(state)

        if drowsiness <= 0 and distraction <= 0:
            self.current_alert = AlertLevel.NONE
            return self.current_alert

        if (now - self.last_alert_time).total_seconds() < self.cooldown_seconds:
            return self.current_alert

        if drowsiness > self.limits["drowsiness_critical"]:
            return self._trigger(AlertLevel.CRITICAL_DROWSINESS, now)

        if distraction > self.limits["distraction_critical"]:
            return self._trigger(AlertLevel.CRITICAL_DISTRACTION, now)

        if drowsiness > self.limits["drowsiness_warning"]:
            return self._trigger(AlertLevel.DROWSINESS_WARNING, now)

        if distraction > self.limits["distraction_warning"]:
            return self._trigger(AlertLevel.DISTRACTION_WARNING, now)

        self.current_alert = AlertLevel.NONE
        return self.current_alert

    def _trigger(self, level: AlertLevel, now: dt.datetime) -> AlertLevel:
        """Sets the current alert level and records the trigger time and triggers the on_alert callback

        Args:
            level (AlertLevel): The alert level to trigger.
            now (dt.datetime): The current timestamp.

        Returns:
            AlertLevel: The triggered alert level.
        """
        if self.current_alert != level:
            self.current_alert = level
            self.last_alert_time = now
            if self.on_alert is not None:
                self.on_alert(level)
        return self.current_alert
