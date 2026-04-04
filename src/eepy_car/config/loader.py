import json
from datetime import datetime
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Function to load configuration from a given file path

    Args:
        path (str | Path): The path to the config.json file

    Returns:
        dict: A dictionary containing the data held within the config file
    """

    try:
        with open(path) as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {path}")

    date_str = datetime.now().strftime("%Y-%m-%d")
    config["output"]["log_path"] = config["output"]["log_path"].replace("{date}", date_str)

    return config
