"""
Carbon data models - normalized internal representation.
Independent of any specific provider's API format.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
import json


class DataSource(Enum):
    """Source of carbon intensity data."""
    UK_CARBON_INTENSITY = "uk_carbon_intensity"
    ELECTRICITY_MAPS = "electricity_maps"
    REPLAY = "replay"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class DataMode(Enum):
    """Operational mode of the carbon data provider."""
    LIVE = "live"
    REPLAY = "replay"
    OFFLINE = "offline"


class ReadingType(Enum):
    """Type of carbon intensity reading."""
    ACTUAL = "actual"
    FORECAST = "forecast"
    HISTORICAL = "historical"


@dataclass
class CarbonReading:
    """
    Normalized carbon intensity reading.
    
    All providers must map their data to this format.
    """
    timestamp: datetime                    # Start of the reading period
    zone: str                              # Geographic zone/region identifier
    carbon_intensity_gco2_per_kwh: float   # gCO2eq/kWh
    data_source: DataSource                # Origin of the data
    reading_type: ReadingType              # actual, forecast, or historical
    horizon_hours: Optional[float] = None  # For forecasts: hours ahead
    is_replay: bool = False                # True if from replay data
    is_offline: bool = False               # True if from offline cache
    generation_mix: Optional[Dict[str, float]] = None  # Optional: fuel mix percentages
    metadata: Dict[str, Any] = field(default_factory=dict)  # Provider-specific extras
    fetched_at: Optional[datetime] = None  # When this reading was fetched from provider
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "zone": self.zone,
            "carbon_intensity_gco2_per_kwh": self.carbon_intensity_gco2_per_kwh,
            "data_source": self.data_source.value,
            "reading_type": self.reading_type.value,
            "horizon_hours": self.horizon_hours,
            "is_replay": self.is_replay,
            "is_offline": self.is_offline,
            "generation_mix": self.generation_mix,
            "metadata": self.metadata,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }
    
    def staleness_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        """
        Calculate data staleness in seconds.
        
        For actual readings: time since the reading period started.
        For forecast readings: time since the forecast was generated.
        For offline fallbacks: returns None (assumed value, not a live observation).
        
        Returns None if timestamps are unavailable or for offline fallbacks.
        """
        # Offline fallbacks are assumed values, not live observations
        if self.is_offline and self.metadata.get("fallback") is True:
            return None
        
        if now is None:
            now = datetime.now(timezone.utc)
        
        # Ensure both are timezone-aware
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        
        return (now - ts).total_seconds()
    
    def is_stale(self, threshold_seconds: float = 3600, now: Optional[datetime] = None) -> Optional[bool]:
        """
        Check if the reading is stale based on a threshold.
        
        Args:
            threshold_seconds: Maximum acceptable age in seconds (default 1 hour)
            now: Reference time (default: current UTC time)
            
        Returns:
            True if stale, False if fresh, None if cannot determine (including offline fallbacks)
        """
        staleness = self.staleness_seconds(now)
        if staleness is None:
            return None
        return staleness > threshold_seconds
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarbonReading":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            zone=data["zone"],
            carbon_intensity_gco2_per_kwh=data["carbon_intensity_gco2_per_kwh"],
            data_source=DataSource(data["data_source"]),
            reading_type=ReadingType(data["reading_type"]),
            horizon_hours=data.get("horizon_hours"),
            is_replay=data.get("is_replay", False),
            is_offline=data.get("is_offline", False),
            generation_mix=data.get("generation_mix"),
            metadata=data.get("metadata", {}),
            fetched_at=datetime.fromisoformat(data["fetched_at"].replace("Z", "+00:00")) if data.get("fetched_at") else None,
        )


@dataclass
class CarbonForecast:
    """Collection of forecast readings for a zone."""
    zone: str
    generated_at: datetime
    readings: List[CarbonReading]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.zone,
            "generated_at": self.generated_at.isoformat(),
            "readings": [r.to_dict() for r in self.readings]
        }


@dataclass
class RegionalCarbonSnapshot:
    """Snapshot of carbon intensity across multiple regions at a given time."""
    timestamp: datetime
    readings: Dict[str, CarbonReading]  # zone -> reading
    mode: DataMode
    
    def get_zones(self) -> List[str]:
        return list(self.readings.keys())
    
    def get_lowest_carbon_zone(self) -> Optional[str]:
        if not self.readings:
            return None
        return min(self.readings.keys(), key=lambda z: self.readings[z].carbon_intensity_gco2_per_kwh)
    
    def get_highest_carbon_zone(self) -> Optional[str]:
        if not self.readings:
            return None
        return max(self.readings.keys(), key=lambda z: self.readings[z].carbon_intensity_gco2_per_kwh)


def calculate_co2_grams(energy_kwh: float, carbon_intensity_gco2_per_kwh: float) -> float:
    """
    Calculate CO2 emissions in grams.
    
    CO2 (g) = energy (kWh) × carbon_intensity (gCO2eq/kWh)
    
    Args:
        energy_kwh: Energy consumption in kilowatt-hours
        carbon_intensity_gco2_per_kwh: Carbon intensity in gCO2eq/kWh
        
    Returns:
        CO2 emissions in grams
    """
    return energy_kwh * carbon_intensity_gco2_per_kwh


def calculate_co2_mg(energy_kwh: float, carbon_intensity_gco2_per_kwh: float) -> float:
    """Calculate CO2 emissions in milligrams."""
    return calculate_co2_grams(energy_kwh, carbon_intensity_gco2_per_kwh) * 1000


def calculate_co2_kg(energy_kwh: float, carbon_intensity_gco2_per_kwh: float) -> float:
    """Calculate CO2 emissions in kilograms."""
    return calculate_co2_grams(energy_kwh, carbon_intensity_gco2_per_kwh) / 1000


# UK region mapping for simulated regional routing
UK_REGIONS = {
    1: "North Scotland",
    2: "South Scotland", 
    3: "North West England",
    4: "North East England",
    5: "Yorkshire",
    6: "North Wales & Mersey",
    7: "South Wales",
    8: "West Midlands",
    9: "East Midlands",
    10: "East England",
    11: "South West England",
    12: "South England",
    13: "London",
    14: "South East England",
    15: "Northern Ireland",  # May not always be present
}

# Simulated regions for demo (maps to UK regions)
SIMULATED_REGIONS = {
    "region_a_low_carbon": "North Scotland",      # Typically lowest carbon
    "region_b_medium_carbon": "South West England",  # Medium
    "region_c_high_carbon": "London",             # Typically higher carbon
}

# Reverse mapping for lookup
UK_REGION_BY_NAME = {v: k for k, v in UK_REGIONS.items()}


def parse_uk_timestamp(ts_str: str) -> datetime:
    """Parse UK Carbon Intensity timestamp string to timezone-aware datetime."""
    # Format: "2026-09-25T13:00Z"
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def format_uk_timestamp(dt: datetime) -> str:
    """Format datetime for UK Carbon Intensity API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%MZ")