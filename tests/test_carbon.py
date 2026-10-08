"""
Tests for CarbonGrid carbon data infrastructure.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import json
import time

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.carbon.models import (
    CarbonReading,
    CarbonForecast,
    RegionalCarbonSnapshot,
    DataSource,
    DataMode,
    ReadingType,
    SIMULATED_REGIONS,
    UK_REGIONS,
    calculate_co2_grams,
    calculate_co2_mg,
    calculate_co2_kg,
    parse_uk_timestamp,
)

from carbongrid.carbon.providers import (
    UKCarbonIntensityProvider,
    ReplayCarbonProvider,
    OfflineCarbonProvider,
    MultiProvider,
)

from carbongrid.carbon.calculator import (
    EnergyMeasurement,
    CarbonEstimate,
    estimate_inference_carbon,
    compare_configurations_carbon,
    PHASE05_BASELINE,
)

from carbongrid.carbon.manager import (
    CarbonManager,
    CarbonMode,
    create_carbon_manager,
)


class TestCarbonModels:
    """Test carbon data models."""
    
    def test_calculate_co2_grams(self):
        """Test CO2 calculation in grams."""
        energy = 0.00005  # kWh
        intensity = 400  # gCO2/kWh
        co2 = calculate_co2_grams(energy, intensity)
        assert co2 == 0.02  # 0.00005 * 400 = 0.02
    
    def test_calculate_co2_mg(self):
        """Test CO2 calculation in milligrams."""
        energy = 0.00005
        intensity = 400
        co2 = calculate_co2_mg(energy, intensity)
        assert co2 == 20.0  # 0.02g = 20mg
    
    def test_calculate_co2_kg(self):
        """Test CO2 calculation in kilograms."""
        energy = 1.0
        intensity = 400
        co2 = calculate_co2_kg(energy, intensity)
        assert co2 == 0.4  # 400g = 0.4kg
    
    def test_carbon_reading_serialization(self):
        """Test CarbonReading to/from dict."""
        reading = CarbonReading(
            timestamp=datetime(2026, 9, 25, 13, 0, tzinfo=timezone.utc),
            zone="national",
            carbon_intensity_gco2_per_kwh=150.5,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL,
            horizon_hours=1.5,
            is_replay=True,
            is_offline=False,
            generation_mix={"wind": 50, "solar": 20},
            metadata={"index": "moderate"}
        )
        
        d = reading.to_dict()
        assert d["zone"] == "national"
        assert d["carbon_intensity_gco2_per_kwh"] == 150.5
        assert d["is_replay"] is True
        
        # Round-trip
        reading2 = CarbonReading.from_dict(d)
        assert reading2.zone == reading.zone
        assert reading2.carbon_intensity_gco2_per_kwh == reading.carbon_intensity_gco2_per_kwh
        assert reading2.is_replay == reading.is_replay
    
    def test_parse_uk_timestamp(self):
        """Test UK timestamp parsing."""
        dt = parse_uk_timestamp("2026-09-25T13:00Z")
        assert dt.year == 2026
        assert dt.month == 9
        assert dt.day == 25
        assert dt.hour == 13
        assert dt.tzinfo is not None
    
    def test_simulated_regions_mapping(self):
        """Test simulated regions map to real UK regions."""
        for sim_region, real_region in SIMULATED_REGIONS.items():
            assert real_region in UK_REGIONS.values()


class TestOfflineProvider:
    """Test offline fallback provider."""
    
    def test_offline_current_intensity(self):
        """Test offline provider returns hardcoded values."""
        provider = OfflineCarbonProvider()
        
        reading = provider.get_current_intensity("national")
        assert reading is not None
        assert reading.zone == "national"
        assert reading.carbon_intensity_gco2_per_kwh == 200
        assert reading.is_offline is True
        assert reading.data_source == DataSource.OFFLINE
    
    def test_offline_forecast(self):
        """Test offline provider generates forecast."""
        provider = OfflineCarbonProvider()
        
        forecast = provider.get_forecast("national", 2)
        assert forecast is not None
        assert len(forecast.readings) == 4  # 2 hours * 2 (30-min intervals)
        assert all(r.is_offline for r in forecast.readings)
        assert all(r.reading_type == ReadingType.FORECAST for r in forecast.readings)
    
    def test_offline_regional(self):
        """Test offline regional snapshot."""
        provider = OfflineCarbonProvider()
        
        snapshot = provider.get_regional_snapshot()
        assert snapshot is not None
        assert snapshot.mode == DataMode.OFFLINE
        assert len(snapshot.readings) > 10
        assert "London" in snapshot.readings
        assert "North Scotland" in snapshot.readings


class TestReplayProvider:
    """Test replay provider."""
    
    def test_replay_load_and_read(self):
        """Test loading and reading replay data."""
        # Create temporary replay file
        readings = [
            CarbonReading(
                timestamp=datetime(2026, 9, 25, 13, 0, tzinfo=timezone.utc),
                zone="national",
                carbon_intensity_gco2_per_kwh=150.0,
                data_source=DataSource.REPLAY,
                reading_type=ReadingType.ACTUAL,
                is_replay=True
            ),
            CarbonReading(
                timestamp=datetime(2026, 9, 25, 13, 30, tzinfo=timezone.utc),
                zone="national",
                carbon_intensity_gco2_per_kwh=160.0,
                data_source=DataSource.REPLAY,
                reading_type=ReadingType.ACTUAL,
                is_replay=True
            ),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([r.to_dict() for r in readings], f)
            temp_path = f.name
        
        try:
            provider = ReplayCarbonProvider(temp_path)
            assert len(provider._replay_data) == 2
            
            # Read first
            r1 = provider.get_current_intensity("national")
            assert r1 is not None
            assert r1.carbon_intensity_gco2_per_kwh == 150.0
            assert r1.is_replay is True
            
            # Read second
            r2 = provider.get_current_intensity("national")
            assert r2 is not None
            assert r2.carbon_intensity_gco2_per_kwh == 160.0
            
            # Exhausted
            r3 = provider.get_current_intensity("national")
            assert r3 is None
        finally:
            Path(temp_path).unlink()


class TestCalculator:
    """Test carbon calculator."""
    
    def test_estimate_inference_carbon(self):
        """Test inference CO2 estimation."""
        energy = EnergyMeasurement(
            energy_joules=180_000,  # 0.05 kWh
            energy_wh=50,
            energy_kwh=0.05,
            duration_seconds=5.0,
            avg_power_watts=30.0,
            samples_count=250
        )
        
        reading = CarbonReading(
            timestamp=datetime.now(timezone.utc),
            zone="national",
            carbon_intensity_gco2_per_kwh=200,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL
        )
        
        estimate = estimate_inference_carbon(energy, reading)
        
        assert estimate.energy_kwh == 0.05
        assert estimate.carbon_intensity_gco2_per_kwh == 200
        assert estimate.co2_grams == 10.0  # 0.05 * 200
        assert estimate.co2_milligrams == 10000.0
    
    def test_compare_configurations_carbon(self):
        """Test FP16 vs INT4 comparison."""
        fp16_energy = 0.0488
        int4_energy = 0.0493
        intensity = 200
        
        comparison = compare_configurations_carbon(fp16_energy, int4_energy, intensity)
        
        assert "fp16" in comparison
        assert "int4" in comparison
        assert "difference" in comparison
        assert comparison["fp16"]["co2_grams"] == 9.76  # 0.0488 * 200
        assert comparison["int4"]["co2_grams"] == 9.86  # 0.0493 * 200


class TestCarbonManager:
    """Test carbon manager."""
    
    def test_create_offline_manager(self):
        """Test creating offline manager."""
        manager = create_carbon_manager("offline")
        assert manager.mode == CarbonMode.OFFLINE
        assert isinstance(manager.provider, OfflineCarbonProvider)
    
    def test_create_replay_manager(self):
        """Test creating replay manager."""
        # Create temp replay data
        readings = [
            CarbonReading(
                timestamp=datetime(2026, 9, 25, 13, 0, tzinfo=timezone.utc),
                zone="national",
                carbon_intensity_gco2_per_kwh=150.0,
                data_source=DataSource.REPLAY,
                reading_type=ReadingType.ACTUAL,
                is_replay=True
            ),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([r.to_dict() for r in readings], f)
            temp_path = f.name
        
        try:
            manager = create_carbon_manager("replay", temp_path)
            assert manager.mode == CarbonMode.REPLAY
            assert isinstance(manager.provider, ReplayCarbonProvider)
        finally:
            Path(temp_path).unlink()
    
    def test_offline_manager_status(self):
        """Test offline manager status."""
        manager = create_carbon_manager("offline")
        status = manager.get_status()
        
        assert status["mode"] == "offline"
        assert status["provider"] == "offline"
        assert "available_zones" in status
        assert "baseline_measurements" in status
        assert "fp16" in status["baseline_measurements"]
        assert "int4" in status["baseline_measurements"]
    
    def test_offline_manager_co2_estimate(self):
        """Test offline manager CO2 estimation."""
        manager = create_carbon_manager("offline")
        
        estimate = manager.estimate_inference_co2_simple(
            energy_kwh=0.05,
            zone="national"
        )
        
        assert estimate is not None
        assert estimate.energy_kwh == 0.05
        assert estimate.zone == "national"
        assert estimate.co2_grams == 10.0  # 0.05 * 200 (offline default)
    
    def test_offline_manager_comparison(self):
        """Test offline manager configuration comparison."""
        manager = create_carbon_manager("offline")
        
        comparison = manager.compare_configurations("national")
        
        assert "fp16" in comparison
        assert "int4" in comparison
        assert "difference" in comparison


class TestUKCarbonIntensityProvider:
    """Test UK Carbon Intensity provider (integration test - requires internet)."""
    
    @pytest.mark.integration
    def test_live_current_intensity(self):
        """Test fetching live current intensity."""
        provider = UKCarbonIntensityProvider()
        reading = provider.get_current_intensity("national")
        
        assert reading is not None
        assert reading.zone == "national"
        assert reading.carbon_intensity_gco2_per_kwh > 0
        assert reading.data_source == DataSource.UK_CARBON_INTENSITY
        assert reading.reading_type in [ReadingType.ACTUAL, ReadingType.FORECAST]
    
    @pytest.mark.integration
    def test_live_forecast(self):
        """Test fetching live forecast."""
        provider = UKCarbonIntensityProvider()
        forecast = provider.get_forecast("national", 48)
        
        assert forecast is not None
        assert forecast.zone == "national"
        assert len(forecast.readings) > 0
        assert all(r.reading_type == ReadingType.FORECAST for r in forecast.readings)
        assert all(r.horizon_hours is not None for r in forecast.readings)
    
    @pytest.mark.integration
    def test_live_regional(self):
        """Test fetching live regional data."""
        provider = UKCarbonIntensityProvider()
        snapshot = provider.get_regional_snapshot()
        
        assert snapshot is not None
        assert len(snapshot.readings) > 0
        assert "London" in snapshot.readings or "North Scotland" in snapshot.readings


class TestCarbonProvenance:
    """Test carbon data provenance and staleness features."""
    
    def test_staleness_calculation_fresh(self):
        """Test staleness calculation for fresh reading."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        reading = CarbonReading(
            timestamp=now - timedelta(minutes=5),
            zone="national",
            carbon_intensity_gco2_per_kwh=200,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL,
        )
        
        staleness = reading.staleness_seconds(now)
        assert staleness is not None
        assert 290 <= staleness <= 310  # ~5 minutes
        assert reading.is_stale(3600, now) is False
    
    def test_staleness_calculation_stale(self):
        """Test staleness calculation for stale reading."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        reading = CarbonReading(
            timestamp=now - timedelta(hours=2),
            zone="national",
            carbon_intensity_gco2_per_kwh=200,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL,
        )
        
        staleness = reading.staleness_seconds(now)
        assert staleness is not None
        assert 7100 <= staleness <= 7300  # ~2 hours
        assert reading.is_stale(3600, now) is True
    
    def test_staleness_calculation_forecast(self):
        """Test staleness for forecast reading (future timestamp)."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        reading = CarbonReading(
            timestamp=now + timedelta(hours=1),
            zone="national",
            carbon_intensity_gco2_per_kwh=150,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.FORECAST,
            horizon_hours=1.0,
        )
        
        staleness = reading.staleness_seconds(now)
        assert staleness is not None
        assert staleness < 0  # Future timestamp = negative staleness
        assert reading.is_stale(3600, now) is False  # Forecasts not stale by age
    
    def test_to_dict_includes_fetched_at(self):
        """Test that to_dict includes fetched_at field."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        reading = CarbonReading(
            timestamp=now,
            zone="national",
            carbon_intensity_gco2_per_kwh=200,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL,
            fetched_at=now,
        )
        
        d = reading.to_dict()
        assert "fetched_at" in d
        assert d["fetched_at"] is not None
        assert d["zone"] == "national"
        assert d["carbon_intensity_gco2_per_kwh"] == 200
    
    def test_offline_reading_has_correct_metadata(self):
        """Test offline readings have fallback metadata."""
        provider = OfflineCarbonProvider()
        reading = provider.get_current_intensity("national")
        
        assert reading.is_offline is True
        assert reading.metadata.get("fallback") is True
        assert "note" in reading.metadata
        assert "Assumed offline value" in reading.metadata.get("note", "")
        assert reading.fetched_at is not None
    
    def test_offline_forecast_has_metadata(self):
        """Test offline forecast readings have fallback metadata."""
        provider = OfflineCarbonProvider()
        forecast = provider.get_forecast("national", 1)
        
        assert forecast is not None
        assert len(forecast.readings) == 2
        for r in forecast.readings:
            assert r.is_offline is True
            assert r.metadata.get("fallback") is True
            assert "Assumed offline value" in r.metadata.get("note", "")
            assert r.fetched_at is not None
    
    def test_offline_regional_has_metadata(self):
        """Test offline regional snapshot has fallback metadata."""
        provider = OfflineCarbonProvider()
        snapshot = provider.get_regional_snapshot()
        
        assert snapshot is not None
        assert snapshot.mode == DataMode.OFFLINE
        for zone, reading in snapshot.readings.items():
            assert reading.is_offline is True
            assert reading.metadata.get("fallback") is True
            assert "Assumed offline value" in reading.metadata.get("note", "")
            assert reading.fetched_at is not None

    def test_offline_fallback_staleness_is_unknown(self):
        """Test that offline fallback staleness is None (unknown, not fresh)."""
        provider = OfflineCarbonProvider()
        reading = provider.get_current_intensity("national")
        
        # Offline fallback should have epoch timestamp
        assert reading.timestamp.year == 1970
        assert reading.timestamp.month == 1
        assert reading.timestamp.day == 1
        
        # Staleness should be None for offline fallbacks
        staleness = reading.staleness_seconds()
        assert staleness is None, "Offline fallback staleness should be None, not a number"
        
        # is_stale should also be None for offline fallbacks
        is_stale = reading.is_stale(3600)
        assert is_stale is None, "Offline fallback is_stale should be None, not True/False"
    
    def test_offline_fallback_fetched_at_is_set(self):
        """Test that fetched_at is set to creation time for offline fallbacks."""
        provider = OfflineCarbonProvider()
        before = datetime.now(timezone.utc)
        reading = provider.get_current_intensity("national")
        after = datetime.now(timezone.utc)
        
        assert reading.fetched_at is not None
        # fetched_at should be between before and after
        assert before <= reading.fetched_at <= after
    
    def test_offline_fallback_cannot_be_mistaken_for_fresh_live(self):
        """Test that offline fallback cannot be mistaken for fresh live observation."""
        provider = OfflineCarbonProvider()
        reading = provider.get_current_intensity("national")
        
        # Key indicators that this is NOT a fresh live observation:
        assert reading.is_offline is True
        assert reading.metadata.get("fallback") is True
        assert "Assumed offline value" in reading.metadata.get("note", "")
        assert reading.timestamp.year == 1970  # Epoch timestamp
        assert reading.staleness_seconds() is None  # Unknown staleness
        assert reading.is_stale(3600) is None  # Cannot determine staleness
        
        # Contrast with a fresh live reading
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        live_reading = CarbonReading(
            timestamp=now - timedelta(minutes=5),
            zone="national",
            carbon_intensity_gco2_per_kwh=200,
            data_source=DataSource.UK_CARBON_INTENSITY,
            reading_type=ReadingType.ACTUAL,
        )
        
        # Live reading has recent timestamp and calculable staleness
        assert live_reading.timestamp.year == now.year
        assert live_reading.staleness_seconds(now) is not None
        assert 290 <= live_reading.staleness_seconds(now) <= 310
        assert live_reading.is_stale(3600, now) is False
        
        # But offline fallback has epoch timestamp and unknown staleness
        assert reading.timestamp.year == 1970
        assert reading.staleness_seconds(now) is None
        assert reading.is_stale(3600, now) is None
    
    def test_offline_forecast_staleness_is_unknown(self):
        """Test that offline forecast staleness is also None."""
        provider = OfflineCarbonProvider()
        forecast = provider.get_forecast("national", 1)
        
        assert forecast is not None
        for r in forecast.readings:
            assert r.is_offline is True
            assert r.timestamp.year == 1970
            assert r.staleness_seconds() is None
            assert r.is_stale(3600) is None
    
    def test_offline_regional_staleness_is_unknown(self):
        """Test that offline regional snapshot staleness is also None."""
        provider = OfflineCarbonProvider()
        snapshot = provider.get_regional_snapshot()
        
        assert snapshot is not None
        for zone, reading in snapshot.readings.items():
            assert reading.is_offline is True
            assert reading.timestamp.year == 1970
            assert reading.staleness_seconds() is None
            assert reading.is_stale(3600) is None


class TestCarbonProviderTimeouts:
    """Test carbon provider timeout behavior and fallback."""

    def test_uk_provider_timeout_parameter(self):
        """Test that UK provider accepts separate connect/read timeouts."""
        provider = UKCarbonIntensityProvider(
            connect_timeout=2.0,
            read_timeout=5.0,
        )
        assert provider.connect_timeout == 2.0
        assert provider.read_timeout == 5.0

    def test_uk_provider_backward_compatibility(self):
        """Test that single timeout parameter still works."""
        provider = UKCarbonIntensityProvider(timeout=8.0)
        assert provider.connect_timeout == 8.0
        assert provider.read_timeout == 8.0

    def test_uk_provider_max_total_time(self):
        """Test that max_total_time_seconds is set."""
        provider = UKCarbonIntensityProvider(max_total_time_seconds=20.0)
        assert provider.max_total_time_seconds == 20.0

    def test_multi_provider_time_budget(self):
        """Test that MultiProvider respects total time budget between providers."""
        # Create a mock provider that fails quickly
        class FailingProvider:
            def __init__(self):
                self.data_source = DataSource.OFFLINE
                self.mode = DataMode.OFFLINE
                self.max_total_time_seconds = 1.0
                
            def get_current_intensity(self, zone="national"):
                raise ConnectionError("Simulated failure")
        
        failing_provider = FailingProvider()
        offline = OfflineCarbonProvider()
        
        multi = MultiProvider([failing_provider, offline])
        
        start = time.time()
        result = multi.get_current_intensity("national")
        elapsed = time.time() - start
        
        # Should fail fast and fallback to offline
        assert result is not None
        assert result.is_offline is True
        assert elapsed < 2.0  # Should not wait long

    def test_multi_provider_timeout_fallback_provenance(self):
        """Test that MultiProvider timeout fallback has correct offline provenance."""
        # Create a UK provider with no retries and short timeout
        class TestUKProvider(UKCarbonIntensityProvider):
            def __init__(self):
                super().__init__(connect_timeout=0.001, read_timeout=0.001, max_retries=0)
        
        multi = MultiProvider([
            TestUKProvider(),
            OfflineCarbonProvider(),
        ])
        
        reading = multi.get_current_intensity("national")
        
        # Should fall back to offline
        assert reading is not None
        assert reading.is_offline is True
        assert reading.metadata.get("fallback") is True
        assert "Assumed offline value" in reading.metadata.get("note", "")
        assert reading.timestamp.year == 1970  # Epoch timestamp
        assert reading.staleness_seconds() is None
        assert reading.is_stale(3600) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])