"""
CarbonGrid Carbon Manager - Main interface for carbon-aware operations.

Combines providers, calculator, and simulated regional routing.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum

from carbongrid.carbon.providers import (
    CarbonDataProvider,
    UKCarbonIntensityProvider,
    ReplayCarbonProvider,
    OfflineCarbonProvider,
    MultiProvider,
    DataMode,
)
from carbongrid.carbon.models import (
    CarbonReading,
    CarbonForecast,
    RegionalCarbonSnapshot,
    SIMULATED_REGIONS,
    DataSource,
)
from carbongrid.carbon.calculator import (
    EnergyMeasurement,
    CarbonEstimate,
    estimate_inference_carbon,
    compare_configurations_carbon,
    PHASE05_BASELINE,
)

logger = logging.getLogger(__name__)


class CarbonMode(Enum):
    """CarbonGrid operational mode."""
    LIVE = "live"
    REPLAY = "replay"
    OFFLINE = "offline"


class CarbonManager:
    """
    Main carbon management interface for CarbonGrid.
    
    Provides high-level operations:
    - Get current carbon intensity
    - Get forecast
    - Estimate CO2 for inference
    - Simulated regional routing comparison
    """
    
    def __init__(self, mode: CarbonMode = CarbonMode.LIVE, replay_file: Optional[str] = None,
                 connect_timeout: float = 3.0, read_timeout: float = 7.0, max_total_time_seconds: float = 15.0):
        self.mode = mode
        
        # Initialize providers based on mode
        if mode == CarbonMode.LIVE:
            self.provider = MultiProvider([
                UKCarbonIntensityProvider(
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    max_total_time_seconds=max_total_time_seconds,
                ),
                ReplayCarbonProvider(replay_file) if replay_file else None,
                OfflineCarbonProvider(),
            ])
            # Remove None providers
            self.provider.providers = [p for p in self.provider.providers if p is not None]
        elif mode == CarbonMode.REPLAY:
            self.provider = ReplayCarbonProvider(replay_file)
        else:  # OFFLINE
            self.provider = OfflineCarbonProvider()
        
        self._last_reading: Optional[CarbonReading] = None
        self._last_regional: Optional[RegionalCarbonSnapshot] = None
    
    def get_current_carbon(self, zone: str = "national") -> Optional[CarbonReading]:
        """Get current carbon intensity for a zone."""
        reading = self.provider.get_current_intensity(zone)
        if reading:
            self._last_reading = reading
        return reading
    
    def get_forecast(self, zone: str = "national", hours: int = 48) -> Optional[CarbonForecast]:
        """Get carbon intensity forecast."""
        return self.provider.get_forecast(zone, hours)
    
    def get_regional_snapshot(self) -> Optional[RegionalCarbonSnapshot]:
        """Get current carbon intensity for all regions."""
        snapshot = self.provider.get_regional_snapshot()
        if snapshot:
            self._last_regional = snapshot
        return snapshot
    
    def estimate_inference_co2(
        self,
        energy_measurement: EnergyMeasurement,
        zone: str = "national"
    ) -> Optional[CarbonEstimate]:
        """
        Estimate CO2 emissions for an inference request.
        
        Args:
            energy_measurement: Measured energy from NVML
            zone: Carbon intensity zone to use
            
        Returns:
            CarbonEstimate with MEASURED energy and ESTIMATED CO2
        """
        reading = self.get_current_carbon(zone)
        if not reading:
            return None
        
        return estimate_inference_carbon(energy_measurement, reading)
    
    def estimate_inference_co2_simple(
        self,
        energy_kwh: float,
        zone: str = "national"
    ) -> Optional[CarbonEstimate]:
        """Simple version using just energy value."""
        reading = self.get_current_carbon(zone)
        if not reading:
            return None
        
        from carbongrid.carbon.calculator import estimate_inference_carbon_simple
        return estimate_inference_carbon_simple(
            energy_kwh=energy_kwh,
            carbon_intensity_gco2_per_kwh=reading.carbon_intensity_gco2_per_kwh,
            zone=zone,
            reading_type=reading.reading_type.value,
            timestamp=reading.timestamp.isoformat()
        )
    
    def compare_configurations(self, zone: str = "national") -> Dict[str, Any]:
        """
        Compare FP16 vs INT4 configurations using current carbon intensity.
        
        Uses Phase 0.5 measured baselines.
        """
        reading = self.get_current_carbon(zone)
        intensity = reading.carbon_intensity_gco2_per_kwh if reading else 200
        
        return compare_configurations_carbon(
            PHASE05_BASELINE["fp16"]["avg_energy_kwh"],
            PHASE05_BASELINE["int4"]["avg_energy_kwh"],
            intensity
        )
    
    def get_simulated_regional_comparison(self) -> Dict[str, Any]:
        """
        Get simulated regional carbon comparison for routing decisions.
        
        Returns carbon intensity for simulated regions (maps to real UK regions).
        Clearly labeled as SIMULATED.
        """
        snapshot = self.get_regional_snapshot()
        if not snapshot:
            return {
                "error": "No regional data available",
                "mode": "SIMULATED_REGIONAL_ROUTING",
                "note": "This is simulated routing for demo purposes only"
            }
        
        results = {}
        for sim_region, real_region in SIMULATED_REGIONS.items():
            if real_region in snapshot.readings:
                reading = snapshot.readings[real_region]
                results[sim_region] = {
                    "mapped_to_real_region": real_region,
                    "carbon_intensity_gco2_per_kwh": reading.carbon_intensity_gco2_per_kwh,
                    "timestamp": reading.timestamp.isoformat(),
                    "data_source": reading.data_source.value,
                    "is_replay": reading.is_replay,
                    "is_offline": reading.is_offline
                }
        
        # Find best region
        if results:
            best = min(results.items(), key=lambda x: x[1]["carbon_intensity_gco2_per_kwh"])
            worst = max(results.items(), key=lambda x: x[1]["carbon_intensity_gco2_per_kwh"])
            
            results["_routing_decision"] = {
                "recommended_region": best[0],
                "recommended_intensity": best[1]["carbon_intensity_gco2_per_kwh"],
                "avoid_region": worst[0],
                "avoid_intensity": worst[1]["carbon_intensity_gco2_per_kwh"],
                "potential_reduction_pct": round(
                    (worst[1]["carbon_intensity_gco2_per_kwh"] - best[1]["carbon_intensity_gco2_per_kwh"]) 
                    / worst[1]["carbon_intensity_gco2_per_kwh"] * 100, 1
                )
            }
        
        return {
            "mode": "SIMULATED_REGIONAL_ROUTING",
            "note": "This is simulated routing for demo purposes only. No actual inference is sent to other regions.",
            "regions": results,
            "timestamp": snapshot.timestamp.isoformat(),
            "data_mode": snapshot.mode.value
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current carbon system status."""
        current = self.get_current_carbon("national")
        now = datetime.now(timezone.utc)
        
        current_info = None
        if current:
            staleness = current.staleness_seconds(now)
            is_stale = current.is_stale(3600, now)  # 1 hour default threshold
            
            current_info = {
                "intensity_gco2_per_kwh": current.carbon_intensity_gco2_per_kwh,
                "zone": current.zone,
                "timestamp": current.timestamp.isoformat(),
                "reading_type": current.reading_type.value,
                "is_replay": current.is_replay,
                "is_offline": current.is_offline,
                "data_source": current.data_source.value,
                "fetched_at": current.fetched_at.isoformat() if current.fetched_at else None,
                "staleness_seconds": round(staleness, 1) if staleness is not None else None,
                "is_stale": is_stale,
                "staleness_threshold_seconds": 3600,
                "metadata": current.metadata,
            }
        
        return {
            "mode": self.mode.value,
            "provider": self.provider.data_source.value,
            "provider_mode": self.provider.mode.value,
            "current_carbon": current_info,
            "available_zones": self.provider.available_zones,
            "baseline_measurements": {
                "note": "From Phase 0.5 validation - MEASURED on RTX 2050 4GB",
                "fp16": PHASE05_BASELINE["fp16"],
                "int4": PHASE05_BASELINE["int4"]
            }
        }


def create_carbon_manager(
    mode: str = "live",
    replay_file: Optional[str] = None,
    connect_timeout: float = 3.0,
    read_timeout: float = 7.0,
    max_total_time_seconds: float = 15.0
) -> CarbonManager:
    """
    Factory function to create CarbonManager.
    
    Args:
        mode: "live", "replay", or "offline"
        replay_file: Path to replay data file (for replay mode)
        connect_timeout: Connection timeout in seconds (default 3.0)
        read_timeout: Read timeout in seconds (default 7.0)
        max_total_time_seconds: Total time budget for provider chain (default 15.0)
        
    Returns:
        Configured CarbonManager instance
    """
    mode_map = {
        "live": CarbonMode.LIVE,
        "replay": CarbonMode.REPLAY,
        "offline": CarbonMode.OFFLINE,
    }
    
    carbon_mode = mode_map.get(mode.lower(), CarbonMode.LIVE)
    return CarbonManager(carbon_mode, replay_file, connect_timeout, read_timeout, max_total_time_seconds)