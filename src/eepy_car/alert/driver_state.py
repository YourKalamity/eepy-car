import datetime as dt


class DriverState:
    """A class that holds the scores for each of the datapoints being tracked
    """
    def __init__(self, config: dict) -> None:
        """Creates an instance of DriverState

        Args:
            config (dict): The configuration dictionary loaded from file
        """
        self.ear_score = 0
        self.mar_score = 0
        self.yaw_score = 0
        self.pitch_score = 0
        self.thresholds = config["thresholds"]
        self.last_t = dt.datetime.now()

    def update_scores(self,
                      ear: float | None,
                      mar: float | None,
                      yaw: float | None,
                      pitch: float | None,
                      current_t: dt.datetime) -> None:
        """Function that takes in the values of the driver's current state and updates the scores
        If data passed is None, then it decays that value safely

        Args:
            ear (float): Eye aspect ratio value
            mar (float): Mouth aspect ratio value
            yaw (float): Head pose yaw offset from headrest tag
            pitch (float): Head pose pitch offset from headrest tag
            current_t (dt.datetime): The current time
        """
        t_delta = (current_t - self.last_t).total_seconds()

        if ear is not None:
            self.ear_score = self.accumulate_when_below(self.ear_score,
                                                        ear,
                                                        self.thresholds["ear"],
                                                        t_delta)
        else:
            self.ear_score = max(0.0, self.ear_score - self.thresholds["ear"]["decay_rate"] * t_delta)

        if mar is not None:
            self.mar_score = self.accumulate_when_above(self.mar_score,
                                                        mar,
                                                        self.thresholds["mar"],
                                                        t_delta)
        else:
            self.mar_score = max(0.0, self.mar_score - self.thresholds["mar"]["decay_rate"] * t_delta)

        if yaw is not None:
            self.yaw_score = self.accumulate_when_above(self.yaw_score,
                                                        abs(yaw),
                                                        self.thresholds["yaw_degrees"],
                                                        t_delta)
        else:
            self.yaw_score = max(0.0, self.yaw_score - self.thresholds["yaw_degrees"]["decay_rate"] * t_delta)

        if pitch is not None:
            self.pitch_score = self.accumulate_when_below(self.pitch_score,
                                                          pitch,
                                                          self.thresholds["pitch_down_degrees"],
                                                          t_delta)
        else:
            self.pitch_score = max(0.0, self.pitch_score - self.thresholds["pitch_down_degrees"]["decay_rate"] * t_delta)

        self.last_t = current_t

    def accumulate_when_below(self,
                              current_score: float,
                              value: float,
                              threshold_dict: dict,
                              t_delta: float) -> float:
        """Helper function that calculates the updated score based on the current value, threshold and decay rate

        Args:
            current_score (float): The current score before processing
            value (float): The value to be compared against the threshold
            threshold_dict (dict): The dictionary holding the threshold information
            t_delta (float): The seconds since the last time the score was updated

        Returns:
            float: The updated score after accumulation or decay.
        """
        threshold = threshold_dict["value"]
        decay = threshold_dict["decay_rate"]
        if value < threshold:
            return current_score + ((threshold - value) * t_delta)

        return max(0.0, current_score - decay * t_delta)

    def accumulate_when_above(self,
                              current_score: float,
                              value: float,
                              threshold_dict: dict,
                              t_delta: float) -> float:
        """Helper function that calculates the updated score based on the current value, threshold and decay rate

        Args:
            current_score (float): The current score before processing
            value (float): The value to be compared against the threshold
            threshold_dict (dict): The dictionary holding the threshold information
            t_delta (float): The seconds since the last time the score was updated

        Returns:
            float: The updated score after accumulation or decay.
        """
        threshold = threshold_dict["value"]
        decay = threshold_dict["decay_rate"]
        if value > threshold:
            return current_score + (value - threshold) * t_delta

        return max(0.0, current_score - decay * t_delta)
