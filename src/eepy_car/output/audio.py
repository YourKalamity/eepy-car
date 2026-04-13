import threading
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from eepy_car.alert import AlertLevel


_AUDIO_KEYS = {
    AlertLevel.DROWSINESS_WARNING: "drowsiness_warning",
    AlertLevel.CRITICAL_DROWSINESS: "drowsiness_critical",
    AlertLevel.DISTRACTION_WARNING: "distraction_warning",
    AlertLevel.CRITICAL_DISTRACTION: "distraction_critical",
}


def play_alert(alert_level: AlertLevel, prev_alert_level: AlertLevel, config: dict) -> None:
    """Plays the audio alert for the given alert level in a daemon thread.

    Args:
        alert_level (AlertLevel): The alert level to play audio for.
        prev_alert_level (AlertLevel): The previous alert level to prevent alerts when alert level downgraded.
        config (dict): The configuration dictionary.
    """
    if not config["output"].get("audio_alert", False):
        return

    if alert_level is AlertLevel.DROWSINESS_WARNING and prev_alert_level is AlertLevel.CRITICAL_DROWSINESS:
        return

    if alert_level is AlertLevel.DISTRACTION_WARNING and prev_alert_level is AlertLevel.CRITICAL_DISTRACTION:
        return

    key = _AUDIO_KEYS.get(alert_level)
    if key is None:
        return

    path = config["output"]["audio"].get(key)
    if path is None or not Path(path).exists():
        return

    thread = threading.Thread(
        target=_play,
        args=(path,),
        daemon=True
    )
    thread.start()


def _play(path: str) -> None:
    """Loads and plays an audio file.

    Args:
        path (str): Path to the audio file.
    """
    try:
        data, samplerate = sf.read(path)
        sd.play(data, samplerate)
        sd.wait()
    except Exception:
        pass
