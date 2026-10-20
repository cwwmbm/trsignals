import tempfile
from pathlib import Path

# Tests never read the user's saved leverage settings or the real vehicle-price cache.
from api import leverage_settings, vehicle_data

_scratch = Path(tempfile.mkdtemp(prefix="trading-tests-"))
leverage_settings.DEFAULT_STORE_PATH = _scratch / "settings.json"
vehicle_data.CACHE_DIR = _scratch / "vehicle"
