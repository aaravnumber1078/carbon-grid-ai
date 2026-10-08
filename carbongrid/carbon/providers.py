"""
Carbon data provider interface and UK Carbon Intensity implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import logging
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from carbongrid.carbon.models import (
    CarbonReading,
    CarbonForecast,
    RegionalCarbonSnapshot,
    DataSource,
    DataMode,
    ReadingType,
    UK_REGIONS,
    UK_REGION_BY_NAME,
    SIMULATED_REGIONS,
    parse_uk_timestamp,
    format_uk_timestamp,
    calculate_co2_grams,
)

logger = logging.getLogger(__name__)


class CarbonDataProvider(ABC):
    """
    Abstract base class for carbon intensity data providers.
    
    All providers must implement these methods to provide a consistent interface.
    """
    
    @abstractmethod
    def get_current_intensity(self, zone: str = "national") -> Optional[CarbonReading]:
        """Get current carbon intensity for a zone."""
        pass
    
    @abstractmethod
    def get_forecast(self, zone: str = "national", hours: int = 48) -> Optional[CarbonForecast]:
        """Get carbon intensity forecast for a zone."""
        pass
    
    @abstractmethod
    def get_historical(self, zone: str, start: datetime, end: datetime) -> List[CarbonReading]:
        """Get historical carbon intensity for a zone in a time range."""
        pass
    
    @abstractmethod
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        """Get current carbon intensity for all available regions."""
        pass
    
    @property
    @abstractmethod
    def data_source(self) -> DataSource:
        """Return the data source identifier."""
        pass
    
    @property
    @abstractmethod
    def mode(self) -> DataMode:
        """Return current operational mode."""
        pass
    
    @property
    @abstractmethod
    def available_zones(self) -> List[str]:
        """Return list of available zones/regions."""
        pass


class UKCarbonIntensityProvider(CarbonDataProvider):
    """
    UK Carbon Intensity API provider.
    
    Free, no API key required.
    Provides: current intensity, 48h forecast, regional data.
    Coverage: UK national + 14 DNO regions.
    """
    
    BASE_URL = "https://api.carbonintensity.org.uk"
    
    def __init__(
        self,
        timeout: int = 10,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        cache_ttl_seconds: int = 300,
        max_total_time_seconds: Optional[float] = None,
        max_retries: int = 3,
    ):
        # Backward compatibility: if connect_timeout/read_timeout not provided, use timeout
        self.connect_timeout = connect_timeout if connect_timeout is not None else timeout
        self.read_timeout = read_timeout if read_timeout is not None else timeout
        self.max_total_time_seconds = max_total_time_seconds
        self.max_retries = max_retries
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Dict[str, tuple[datetime, Any]] = {}
        
        # Session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self._mode = DataMode.LIVE
    
    @property
    def data_source(self) -> DataSource:
        return DataSource.UK_CARBON_INTENSITY
    
    @property
    def mode(self) -> DataMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: DataMode):
        self._mode = value
    
    @property
    def available_zones(self) -> List[str]:
        return ["national"] + list(UK_REGIONS.values())
    
    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self._cache:
            cached_time, data = self._cache[key]
            if datetime.now(timezone.utc) - cached_time < self.cache_ttl:
                return data
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (datetime.now(timezone.utc), data)
    
    def _make_request(self, endpoint: str, deadline: Optional[float] = None) -> Optional[Dict]:
        """Make HTTP request with error handling and deadline awareness.
        
        Args:
            endpoint: API endpoint to request
            deadline: Absolute deadline timestamp (from time.time()), or None for no deadline
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Calculate max retries based on deadline
        max_retries = self.max_retries
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning(f"Deadline exceeded before request to {url}")
                return None
            # Estimate max time per attempt (connect + read + small buffer)
            max_per_attempt = self.connect_timeout + self.read_timeout + 1.0
            # Calculate how many attempts fit in remaining time (with backoff)
            # backoff times: 1s, 2s, 4s... (exponential with factor 1)
            estimated_total = 0
            attempts = 0
            for i in range(self.max_retries + 1):
                attempt_time = self.connect_timeout + self.read_timeout
                if i > 0:
                    attempt_time += min(2 ** (i - 1), 10)  # backoff: 1, 2, 4, 8...
                if estimated_total + attempt_time > deadline - time.time() - 1.0:
                    break
                estimated_total += attempt_time
                attempts += 1
            max_retries = max(0, attempts - 1)
            if max_retries < self.max_retries:
                logger.warning(f"Reducing retries from {self.max_retries} to {max_retries} to meet deadline")
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        try:
            # Use separate connect and read timeouts
            timeout = (self.connect_timeout, self.read_timeout)
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout requesting {url}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error for {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error for {url}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error for {url}: {e}")
        return None
    
    def get_current_intensity(self, zone: str = "national", deadline: Optional[float] = None) -> Optional[CarbonReading]:
        """Get current carbon intensity (national or regional)."""
        if self._mode == DataMode.OFFLINE:
            return self._get_offline_current(zone)
        
        cache_key = f"current_{zone}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if zone == "national":
            data = self._make_request("/intensity", deadline=deadline)
            if not data:
                return self._get_offline_current(zone)
            
            reading = self._parse_national_reading(data["data"][0], ReadingType.ACTUAL)
        else:
            # Regional - need to find region ID
            region_id = UK_REGION_BY_NAME.get(zone)
            if not region_id:
                logger.warning(f"Unknown zone: {zone}")
                return None
            
            data = self._make_request("/regional", deadline=deadline)
            if not data:
                return self._get_offline_current(zone)
            
            reading = self._parse_regional_reading(data["data"][0], region_id, ReadingType.ACTUAL)
        
        if reading:
            self._set_cache(cache_key, reading)
        return reading
    
    def get_forecast(self, zone: str = "national", hours: int = 48, deadline: Optional[float] = None) -> Optional[CarbonForecast]:
        """Get carbon intensity forecast (up to 48 hours)."""
        if self._mode == DataMode.OFFLINE:
            return self._get_offline_forecast(zone, hours)
        
        cache_key = f"forecast_{zone}_{hours}h"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        now = datetime.now(timezone.utc)
        # API expects format: /intensity/{from}/fw{hours}h
        from_str = format_uk_timestamp(now)
        endpoint = f"/intensity/{from_str}/fw{hours}h"
        
        data = self._make_request(endpoint, deadline=deadline)
        if not data:
            return self._get_offline_forecast(zone, hours)
            return self._get_offline_forecast(zone, hours)
        
        readings = []
        for item in data.get("data", []):
            reading = self._parse_national_reading(item, ReadingType.FORECAST)
            if reading:
                # Calculate horizon
                horizon = (reading.timestamp - now).total_seconds() / 3600
                reading.horizon_hours = round(horizon, 1)
                readings.append(reading)
        
        forecast = CarbonForecast(
            zone=zone,
            generated_at=now,
            readings=readings
        )
        self._set_cache(cache_key, forecast)
        return forecast
    
    def get_historical(self, zone: str, start: datetime, end: datetime) -> List[CarbonReading]:
        """Get historical data - UK API doesn't directly support arbitrary historical queries."""
        # For historical data, we'd need to use replay/offline data
        # This provider focuses on live + forecast
        logger.warning("Historical queries not directly supported by UK API. Use replay mode.")
        return []
    
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        """Get current carbon intensity for all UK regions."""
        if self._mode == DataMode.OFFLINE:
            return self._get_offline_regional()
        
        cache_key = "regional_snapshot"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._make_request("/regional")
        if not data:
            return self._get_offline_regional()
        
        readings = {}
        period_data = data["data"][0]
        timestamp = parse_uk_timestamp(period_data["from"])
        
        for region_data in period_data.get("regions", []):
            region_id = region_data["regionid"]
            zone_name = UK_REGIONS.get(region_id)
            if zone_name:
                reading = self._parse_regional_reading(period_data, region_id, ReadingType.ACTUAL)
                if reading:
                    readings[zone_name] = reading
        
        snapshot = RegionalCarbonSnapshot(
            timestamp=timestamp,
            readings=readings,
            mode=self._mode
        )
        self._set_cache(cache_key, snapshot)
        return snapshot
    
    def _parse_national_reading(self, data: Dict, reading_type: ReadingType) -> Optional[CarbonReading]:
        """Parse national intensity data."""
        try:
            intensity_data = data["intensity"]
            # Prefer actual over forecast for ACTUAL type
            if reading_type == ReadingType.ACTUAL:
                intensity = intensity_data.get("actual")
                if intensity is None:
                    intensity = intensity_data.get("forecast")
            else:
                intensity = intensity_data.get("forecast")
            
            if intensity is None:
                return None
            
            return CarbonReading(
                timestamp=parse_uk_timestamp(data["from"]),
                zone="national",
                carbon_intensity_gco2_per_kwh=float(intensity),
                data_source=self.data_source,
                reading_type=reading_type,
                metadata={"index": intensity_data.get("index")},
                fetched_at=datetime.now(timezone.utc)
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse national reading: {e}")
            return None
    
    def _parse_regional_reading(self, period_data: Dict, region_id: int, reading_type: ReadingType) -> Optional[CarbonReading]:
        """Parse regional intensity data."""
        try:
            zone_name = UK_REGIONS.get(region_id)
            if not zone_name:
                return None
            
            # Find region in period data
            region_data = None
            for r in period_data.get("regions", []):
                if r["regionid"] == region_id:
                    region_data = r
                    break
            
            if not region_data:
                return None
            
            intensity_data = region_data["intensity"]
            intensity = intensity_data.get("forecast")  # Regional usually only has forecast
            
            if intensity is None:
                return None
            
            return CarbonReading(
                timestamp=parse_uk_timestamp(period_data["from"]),
                zone=zone_name,
                carbon_intensity_gco2_per_kwh=float(intensity),
                data_source=self.data_source,
                reading_type=reading_type,
                metadata={
                    "index": intensity_data.get("index"),
                    "region_id": region_id,
                    "dnoregion": region_data.get("dnoregion")
                },
                fetched_at=datetime.now(timezone.utc)
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse regional reading: {e}")
            return None
    
    # Offline/replay fallback methods
    def _get_offline_current(self, zone: str) -> Optional[CarbonReading]:
        """Get current reading from offline cache."""
        # Will be implemented with replay data loader
        return None
    
    def _get_offline_forecast(self, zone: str, hours: int) -> Optional[CarbonForecast]:
        return None
    
    def _get_offline_regional(self) -> Optional[RegionalCarbonSnapshot]:
        return None
    
    def set_replay_data(self, replay_data: List[CarbonReading]):
        """Inject replay data for offline/replay mode."""
        self._replay_data = replay_data
        self._mode = DataMode.REPLAY


class ReplayCarbonProvider(CarbonDataProvider):
    """
    Replay/offline carbon data provider.
    
    Loads pre-recorded carbon intensity data from files.
    Used for testing, demos, and when live API is unavailable.
    """
    
    def __init__(self, replay_file: Optional[str] = None):
        self._replay_data: List[CarbonReading] = []
        self._index = 0
        self._mode = DataMode.REPLAY
        self._replay_file = replay_file
        if replay_file:
            self.load_replay_data(replay_file)
    
    @property
    def data_source(self) -> DataSource:
        return DataSource.REPLAY
    
    @property
    def mode(self) -> DataMode:
        return self._mode
    
    @property
    def available_zones(self) -> List[str]:
        zones = set()
        for r in self._replay_data:
            zones.add(r.zone)
        return sorted(zones)
    
    def load_replay_data(self, filepath: str) -> int:
        """Load replay data from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._replay_data = [CarbonReading.from_dict(item) for item in data]
        self._index = 0
        logger.info(f"Loaded {len(self._replay_data)} replay readings from {filepath}")
        return len(self._replay_data)
    
    def get_current_intensity(self, zone: str = "national", deadline: Optional[float] = None) -> Optional[CarbonReading]:
        """Get next reading from replay data for the zone."""
        for i in range(self._index, len(self._replay_data)):
            reading = self._replay_data[i]
            if reading.zone == zone:
                self._index = i + 1
                reading.is_replay = True
                return reading
        return None
    
    def get_forecast(self, zone: str = "national", hours: int = 48, deadline: Optional[float] = None) -> Optional[CarbonForecast]:
        """Generate forecast from replay data (next N readings)."""
        zone_readings = [r for r in self._replay_data if r.zone == zone]
        if not zone_readings:
            return None
        
        # Take next N readings as "forecast"
        forecast_readings = zone_readings[self._index:self._index + hours * 2]  # 30-min intervals
        for r in forecast_readings:
            r.reading_type = ReadingType.FORECAST
            r.is_replay = True
        
        return CarbonForecast(
            zone=zone,
            generated_at=datetime.now(timezone.utc),
            readings=forecast_readings
        )
    
    def get_historical(self, zone: str, start: datetime, end: datetime) -> List[CarbonReading]:
        """Get historical readings from replay data in time range."""
        results = []
        for r in self._replay_data:
            if r.zone == zone and start <= r.timestamp <= end:
                results.append(r)
        return results
    
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        """Get regional snapshot from replay data at current index."""
        # Group readings at current timestamp
        if self._index >= len(self._replay_data):
            return None
        
        current_time = self._replay_data[self._index].timestamp
        readings = {}
        
        for r in self._replay_data:
            if r.timestamp == current_time:
                readings[r.zone] = r
        
        if not readings:
            return None
        
        return RegionalCarbonSnapshot(
            timestamp=current_time,
            readings=readings,
            mode=DataMode.REPLAY
        )
    
    def reset(self):
        """Reset replay to beginning."""
        self._index = 0


class OfflineCarbonProvider(CarbonDataProvider):
    """
    Minimal offline provider with hardcoded fallback data.
    
    Used when no internet and no replay data available.
    """
    
    def __init__(self):
        self._mode = DataMode.OFFLINE
        # Default fallback values (UK average-ish)
        self._fallback_data = {
            "national": 200,
            "North Scotland": 50,
            "South Scotland": 80,
            "North West England": 180,
            "Yorkshire": 220,
            "South Wales": 250,
            "West Midlands": 280,
            "East Midlands": 260,
            "East England": 240,
            "South West England": 150,
            "South England": 230,
            "London": 300,
            "South East England": 270,
        }
    
    @property
    def data_source(self) -> DataSource:
        return DataSource.OFFLINE
    
    @property
    def mode(self) -> DataMode:
        return self._mode
    
    @property
    def available_zones(self) -> List[str]:
        return list(self._fallback_data.keys())
    
    def get_current_intensity(self, zone: str = "national", deadline: Optional[float] = None) -> Optional[CarbonReading]:
        intensity = self._fallback_data.get(zone, 200)
        now = datetime.now(timezone.utc)
        # Use epoch as timestamp to clearly indicate this is NOT a live observation
        fallback_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return CarbonReading(
            timestamp=fallback_timestamp,
            zone=zone,
            carbon_intensity_gco2_per_kwh=float(intensity),
            data_source=self.data_source,
            reading_type=ReadingType.ACTUAL,
            is_offline=True,
            metadata={"fallback": True, "note": "Assumed offline value - not a live measurement"},
            fetched_at=now
        )
    
    def get_forecast(self, zone: str = "national", hours: int = 48, deadline: Optional[float] = None) -> Optional[CarbonForecast]:
        base = self._fallback_data.get(zone, 200)
        now = datetime.now(timezone.utc)
        fallback_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
        readings = []
        for h in range(hours * 2):  # 30-min intervals
            readings.append(CarbonReading(
                timestamp=fallback_timestamp,
                zone=zone,
                carbon_intensity_gco2_per_kwh=float(base),
                data_source=self.data_source,
                reading_type=ReadingType.FORECAST,
                horizon_hours=h * 0.5,
                is_offline=True,
                metadata={"fallback": True, "note": "Assumed offline value - not a live measurement"},
                fetched_at=now
            ))
        return CarbonForecast(zone=zone, generated_at=now, readings=readings)
    
    def get_historical(self, zone: str, start: datetime, end: datetime) -> List[CarbonReading]:
        return []
    
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        now = datetime.now(timezone.utc)
        fallback_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
        readings = {}
        for zone, intensity in self._fallback_data.items():
            readings[zone] = CarbonReading(
                timestamp=fallback_timestamp,
                zone=zone,
                carbon_intensity_gco2_per_kwh=float(intensity),
                data_source=self.data_source,
                reading_type=ReadingType.ACTUAL,
                is_offline=True,
                metadata={"fallback": True, "note": "Assumed offline value - not a live measurement"},
                fetched_at=now
            )
        return RegionalCarbonSnapshot(
            timestamp=now,
            readings=readings,
            mode=DataMode.OFFLINE
        )


class MultiProvider(CarbonDataProvider):
    """
    Composite provider that tries multiple providers in order.
    
    Falls back from LIVE -> REPLAY -> OFFLINE automatically.
    """
    
    def __init__(self, providers: List[CarbonDataProvider]):
        self.providers = providers
        self._current_provider_idx = 0
    
    @property
    def data_source(self) -> DataSource:
        return self.providers[self._current_provider_idx].data_source
    
    @property
    def mode(self) -> DataMode:
        return self.providers[self._current_provider_idx].mode
    
    @property
    def available_zones(self) -> List[str]:
        # Union of all zones
        zones = set()
        for p in self.providers:
            zones.update(p.available_zones)
        return sorted(zones)
    
    def _try_providers(self, method_name: str, *args, deadline: Optional[float] = None, **kwargs) -> Any:
        """Try providers in order with a total time budget.
        
        If a provider has a max_total_time_seconds attribute, it will be used as the
        overall budget. Otherwise, the first provider's max_total_time_seconds (if any)
        is used. If no budget is set, there is no limit.
        """
        # Determine total time budget
        total_budget = None
        if self.providers:
            first_provider = self.providers[0]
            if hasattr(first_provider, 'max_total_time_seconds') and first_provider.max_total_time_seconds:
                total_budget = first_provider.max_total_time_seconds
        
        # Compute deadline
        deadline = None
        if total_budget is not None:
            deadline = time.time() + total_budget
        
        start_time = time.time()
        
        for i, provider in enumerate(self.providers):
            # Check time budget before each attempt
            if total_budget is not None:
                elapsed = time.time() - start_time
                if elapsed >= total_budget:
                    logger.warning(f"Total time budget ({total_budget}s) exceeded after {elapsed:.1f}s. Stopping provider chain.")
                    return None
            
            try:
                method = getattr(provider, method_name)
                # Pass deadline to provider methods that support it
                if method_name in ("get_current_intensity", "get_forecast"):
                    result = method(*args, deadline=deadline, **kwargs)
                else:
                    result = method(*args, **kwargs)
                if result is not None:
                    if i != self._current_provider_idx:
                        logger.info(f"Falling back to provider {i}: {provider.data_source.value}")
                        self._current_provider_idx = i
                    return result
            except requests.exceptions.Timeout:
                logger.warning(f"Provider {provider.data_source.value} timed out")
                continue
            except Exception as e:
                logger.warning(f"Provider {provider.data_source.value} failed: {e}")
                continue
        return None
    
    def get_current_intensity(self, zone: str = "national", deadline: Optional[float] = None) -> Optional[CarbonReading]:
        return self._try_providers("get_current_intensity", zone, deadline=deadline)
    
    def get_forecast(self, zone: str = "national", hours: int = 48, deadline: Optional[float] = None) -> Optional[CarbonForecast]:
        return self._try_providers("get_forecast", zone, hours, deadline=deadline)
    
    def get_historical(self, zone: str, start: datetime, end: datetime) -> List[CarbonReading]:
        return self._try_providers("get_historical", zone, start, end) or []
    
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        return self._try_providers("get_regional_snapshot")