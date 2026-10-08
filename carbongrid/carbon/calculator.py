"""
Carbon calculation utilities for CarbonGrid.

Handles CO2 estimation from measured energy and carbon intensity.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from carbongrid.carbon.models import (
    CarbonReading,
    calculate_co2_grams,
    calculate_co2_mg,
    calculate_co2_kg,
)


class EnergyUnit(Enum):
    """Energy units for conversion."""
    JOULES = "joules"
    WATT_HOURS = "wh"
    KILOWATT_HOURS = "kwh"
    MILLIWATT_HOURS = "mwh"


@dataclass
class EnergyMeasurement:
    """
    Measured energy consumption from inference.
    
    This comes from NVML power sampling integration.
    """
    energy_joules: float
    energy_wh: float
    energy_kwh: float
    duration_seconds: float
    avg_power_watts: float
    samples_count: int
    
    def to_dict(self) -> dict:
        return {
            "energy_joules": self.energy_joules,
            "energy_wh": self.energy_wh,
            "energy_kwh": self.energy_kwh,
            "duration_seconds": self.duration_seconds,
            "avg_power_watts": self.avg_power_watts,
            "samples_count": self.samples_count
        }


@dataclass
class CarbonEstimate:
    """
    Estimated CO2 emissions from inference.
    
    DISTINCTION:
    - energy_kwh: MEASURED (from NVML power integration)
    - carbon_intensity: FROM PROVIDER (live/replay/offline)
    - co2_grams: ESTIMATED (energy × carbon_intensity)
    """
    energy_kwh: float                      # MEASURED
    carbon_intensity_gco2_per_kwh: float   # FROM PROVIDER
    co2_grams: float                       # ESTIMATED
    co2_milligrams: float                  # ESTIMATED
    zone: str                              # Zone used for carbon intensity
    reading_type: str                      # actual/forecast
    timestamp: str                         # When the carbon intensity was measured
    
    def to_dict(self) -> dict:
        return {
            "energy_kwh": self.energy_kwh,
            "carbon_intensity_gco2_per_kwh": self.carbon_intensity_gco2_per_kwh,
            "co2_grams": self.co2_grams,
            "co2_milligrams": self.co2_milligrams,
            "zone": self.zone,
            "reading_type": self.reading_type,
            "timestamp": self.timestamp,
            "note": "energy=MEASURED, carbon_intensity=PROVIDER, co2=ESTIMATED"
        }


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kilowatt-hours."""
    return joules / 3_600_000  # 1 kWh = 3.6M J


def wh_to_kwh(wh: float) -> float:
    """Convert watt-hours to kilowatt-hours."""
    return wh / 1000


def estimate_inference_carbon(
    energy_measurement: EnergyMeasurement,
    carbon_reading: CarbonReading
) -> CarbonEstimate:
    """
    Estimate CO2 emissions from an inference request.
    
    Args:
        energy_measurement: Measured energy from NVML
        carbon_reading: Carbon intensity from provider
        
    Returns:
        CarbonEstimate with MEASURED energy and ESTIMATED CO2
    """
    co2_g = calculate_co2_grams(energy_measurement.energy_kwh, carbon_reading.carbon_intensity_gco2_per_kwh)
    co2_mg = calculate_co2_mg(energy_measurement.energy_kwh, carbon_reading.carbon_intensity_gco2_per_kwh)
    
    return CarbonEstimate(
        energy_kwh=energy_measurement.energy_kwh,
        carbon_intensity_gco2_per_kwh=carbon_reading.carbon_intensity_gco2_per_kwh,
        co2_grams=co2_g,
        co2_milligrams=co2_mg,
        zone=carbon_reading.zone,
        reading_type=carbon_reading.reading_type.value,
        timestamp=carbon_reading.timestamp.isoformat()
    )


def estimate_inference_carbon_simple(
    energy_kwh: float,
    carbon_intensity_gco2_per_kwh: float,
    zone: str = "unknown",
    reading_type: str = "actual",
    timestamp: Optional[str] = None
) -> CarbonEstimate:
    """Simple version without full objects."""
    from datetime import datetime, timezone
    
    co2_g = calculate_co2_grams(energy_kwh, carbon_intensity_gco2_per_kwh)
    co2_mg = calculate_co2_mg(energy_kwh, carbon_intensity_gco2_per_kwh)
    
    return CarbonEstimate(
        energy_kwh=energy_kwh,
        carbon_intensity_gco2_per_kwh=carbon_intensity_gco2_per_kwh,
        co2_grams=co2_g,
        co2_milligrams=co2_mg,
        zone=zone,
        reading_type=reading_type,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat()
    )


def compare_configurations_carbon(
    energy_fp16_kwh: float,
    energy_int4_kwh: float,
    carbon_intensity_gco2_per_kwh: float
) -> dict:
    """
    Compare CO2 emissions between FP16 and INT4 configurations.
    
    Returns both absolute and relative comparisons.
    """
    co2_fp16 = calculate_co2_grams(energy_fp16_kwh, carbon_intensity_gco2_per_kwh)
    co2_int4 = calculate_co2_grams(energy_int4_kwh, carbon_intensity_gco2_per_kwh)
    
    diff_g = co2_fp16 - co2_int4
    diff_pct = (diff_g / co2_fp16 * 100) if co2_fp16 > 0 else 0
    
    return {
        "fp16": {
            "energy_kwh": energy_fp16_kwh,
            "co2_grams": co2_fp16,
            "co2_milligrams": co2_fp16 * 1000
        },
        "int4": {
            "energy_kwh": energy_int4_kwh,
            "co2_grams": co2_int4,
            "co2_milligrams": co2_int4 * 1000
        },
        "difference": {
            "co2_grams_saved": diff_g,
            "co2_milligrams_saved": diff_g * 1000,
            "percent_reduction": diff_pct,
            "note": "Positive = INT4 emits less CO2"
        },
        "carbon_intensity_used": carbon_intensity_gco2_per_kwh,
        "measurement_note": "energy=MEASURED (Phase 0.5), co2=ESTIMATED"
    }


# Phase 0.5 measured baselines (from validation)
PHASE05_BASELINE = {
    "fp16": {
        "avg_energy_kwh": 0.0488,      # MEASURED
        "avg_latency_ms": 5867,
        "avg_throughput_tok_s": 21.9,
        "avg_power_w": 30.45,
        "peak_vram_mb": 2960,
    },
    "int4": {
        "avg_energy_kwh": 0.0493,      # MEASURED
        "avg_latency_ms": 10580,
        "avg_throughput_tok_s": 12.1,
        "avg_power_w": 17.12,
        "peak_vram_mb": 1161,
    }
}


def get_baseline_comparison(carbon_intensity: float = 200) -> dict:
    """Get baseline comparison using Phase 0.5 measured values."""
    fp16_energy = PHASE05_BASELINE["fp16"]["avg_energy_kwh"]
    int4_energy = PHASE05_BASELINE["int4"]["avg_energy_kwh"]
    
    return compare_configurations_carbon(fp16_energy, int4_energy, carbon_intensity)


def print_carbon_estimate(estimate: CarbonEstimate):
    """Pretty print a carbon estimate."""
    print(f"\nCarbon Estimate:")
    print(f"  Energy (MEASURED):     {estimate.energy_kwh:.6f} kWh ({estimate.energy_kwh*1000:.3f} Wh)")
    print(f"  Carbon Intensity:      {estimate.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh")
    print(f"  CO2 (ESTIMATED):       {estimate.co2_grams:.4f} g = {estimate.co2_milligrams:.2f} mg")
    print(f"  Zone:                  {estimate.zone}")
    print(f"  Reading Type:          {estimate.reading_type}")


def print_comparison(comparison: dict):
    """Pretty print configuration comparison."""
    print(f"\n{'='*50}")
    print(f"FP16 vs INT4 Carbon Comparison")
    print(f"{'='*50}")
    print(f"Carbon Intensity: {comparison['carbon_intensity_used']:.1f} gCO2/kWh")
    print(f"\nFP16 (High Compute):")
    print(f"  Energy:  {comparison['fp16']['energy_kwh']:.6f} kWh")
    print(f"  CO2:     {comparison['fp16']['co2_grams']:.4f} g ({comparison['fp16']['co2_milligrams']:.2f} mg)")
    print(f"\nINT4 (Low Compute):")
    print(f"  Energy:  {comparison['int4']['energy_kwh']:.6f} kWh")
    print(f"  CO2:     {comparison['int4']['co2_grams']:.4f} g ({comparison['int4']['co2_milligrams']:.2f} mg)")
    print(f"\nDifference:")
    diff = comparison['difference']
    print(f"  CO2 Saved:  {diff['co2_grams_saved']:.6f} g ({diff['co2_milligrams_saved']:.3f} mg)")
    print(f"  Reduction:  {diff['percent_reduction']:.2f}%")
    print(f"  Note: {diff['note']}")
    print(f"\n{comparison['measurement_note']}")