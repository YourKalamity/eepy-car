from eepy_car.config import load_config
from eepy_car.capture import CaptureManager

if __name__ == "__main__":
    cur_config = load_config("config.json")
    
    with CaptureManager(cur_config["camera"]["index"]):
        

