import tempfile
from pathlib import Path

# Tests never read the user's saved leverage settings or the real vehicle-price cache.
from api import earnings_data, leverage_settings, vehicle_data

_scratch = Path(tempfile.mkdtemp(prefix="trading-tests-"))
leverage_settings.DEFAULT_STORE_PATH = _scratch / "settings.json"
vehicle_data.CACHE_DIR = _scratch / "vehicle"
# ... and never ask Yahoo for earnings dates: with nothing to fetch, every frame reads "none known".
earnings_data.CACHE_DIR = _scratch / "earnings"


class _NoEarningsTicker:
    def get_earnings_dates(self, limit=12, offset=0):
        return None


earnings_data._ticker = lambda yf_symbol: _NoEarningsTicker()
