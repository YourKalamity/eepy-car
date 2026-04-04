import json
import pytest
from datetime import datetime
from pathlib import Path

from eepy_car.config import load_config


@pytest.fixture
def valid_config(tmp_path):
    """Writes a valid config.json to a temp directory and returns its path."""
    config = {
        "camera": {
            "index": 0
        },
        "thresholds": {
            "ear": 0.20,
            "mar": 0.60,
            "gaze_degrees": 20.0,
            "head_distance_metres": 0.3
        },
        "apriltag": {
            "family": "tag36h11",
            "tag_size_metres": 0.055,
            "headrest_tag_id": 250
        },
        "output": {
            "log_events": True,
            "log_path": "logs/events-{date}.log",
            "show_overlay": True,
            "audio_alert": True
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestLoadConfig:

    def test_loads_successfully(self, valid_config):
        """Should return a dict when given a valid config file"""
        config = load_config(valid_config)
        assert isinstance(config, dict)

    def test_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError if the config file does not exist"""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.json")

    def test_date_placeholder_substituted(self, valid_config):
        """The {date} placeholder in log_path should be replaced with today's date"""
        config = load_config(valid_config)
        expected_date = datetime.now().strftime("%Y-%m-%d")
        assert "{date}" not in config["output"]["log_path"]
        assert expected_date in config["output"]["log_path"]

    def test_log_path_format(self, valid_config):
        """Log path should follow the events-YYYY-MM-DD.log format"""
        config = load_config(valid_config)
        log_path = Path(config["output"]["log_path"])
        assert log_path.suffix == ".log"
        assert log_path.stem.startswith("events-")

    def test_thresholds_present(self, valid_config):
        """All expected threshold keys should be present in the loaded config"""
        config = load_config(valid_config)
        expected_keys = {"ear", "mar", "gaze_degrees", "head_distance_metres"}
        assert expected_keys.issubset(config["thresholds"].keys())

    def test_threshold_types_are_numeric(self, valid_config):
        """Threshold values should all be numeric."""
        config = load_config(valid_config)
        for key, value in config["thresholds"].items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"

    def test_accepts_path_object(self, valid_config):
        """Should accept a Path object as well as a string"""
        config = load_config(Path(valid_config))
        assert isinstance(config, dict)

    def test_accepts_string_path(self, valid_config):
        """Should accept a plain string path"""
        config = load_config(str(valid_config))
        assert isinstance(config, dict)

    def test_raises_on_invalid_json(self, tmp_path):
        """Should raise an error if the file contains invalid JSON"""
        bad_config = tmp_path / "config.json"
        bad_config.write_text("ijasodilasjfoia {{{")
        with pytest.raises(json.JSONDecodeError):
            load_config(bad_config)
