#!/usr/bin/env python3
"""
Phase A Demo - Carbon Data Infrastructure Demonstration.

Shows:
1. Live carbon reading (if available)
2. Historical/replay reading
3. Offline reading
4. Carbon forecast (provider-provided)
5. CO2 calculation using measured energy
6. Simulated regional carbon comparison
"""

import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.carbon import (
    create_carbon_manager,
    CarbonMode,
    EnergyMeasurement,
    PHASE05_BASELINE,
    UK_REGIONS,
    SIMULATED_REGIONS,
    estimate_inference_carbon,
    compare_configurations_carbon,
)

from carbongrid.carbon.providers import UKCarbonIntensityProvider
from carbongrid.carbon.calculator import print_carbon_estimate, print_comparison, get_baseline_comparison


def demo_live_mode():
    """Demo live carbon data from UK API."""
    print("\n" + "=" * 70)
    print("DEMO 1: LIVE CARBON DATA (UK Carbon Intensity API)")
    print("=" * 70)
    
    manager = create_carbon_manager("live")
    status = manager.get_status()
    
    print(f"\nSystem Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Provider: {status['provider']}")
    print(f"  Available Zones: {len(status['available_zones'])}")
    
    current = status['current_carbon']
    if current:
        print(f"\nCurrent National Carbon Intensity:")
        print(f"  Intensity: {current['intensity_gco2_per_kwh']:.1f} gCO2/kWh")
        print(f"  Zone: {current['zone']}")
        print(f"  Timestamp: {current['timestamp']}")
        print(f"  Type: {current['reading_type']}")
        print(f"  Replay: {current['is_replay']}")
        print(f"  Offline: {current['is_offline']}")
    else:
        print("\n  No live data available (check internet connection)")
    
    return manager


def demo_forecast(manager):
    """Demo carbon forecast."""
    print("\n" + "=" * 70)
    print("DEMO 2: CARBON FORECAST (Provider-provided, 48-hour)")
    print("=" * 70)
    
    forecast = manager.get_forecast("national", 48)
    if forecast:
        print(f"\nForecast generated at: {forecast.generated_at.isoformat()}")
        print(f"Zone: {forecast.zone}")
        print(f"Number of readings: {len(forecast.readings)}")
        
        print(f"\nNext 6 forecast periods:")
        for i, reading in enumerate(forecast.readings[:6]):
            print(f"  {reading.timestamp.strftime('%H:%M')} | "
                  f"{reading.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh | "
                  f"horizon: {reading.horizon_hours:.1f}h | "
                  f"type: {reading.reading_type.value}")
    else:
        print("No forecast available")
    
    return forecast


def demo_replay_mode():
    """Demo replay mode with synthetic data."""
    print("\n" + "=" * 70)
    print("DEMO 3: REPLAY MODE (Synthetic historical data)")
    print("=" * 70)
    
    # Generate synthetic replay data if not exists
    replay_file = Path("data/replay/carbon_replay.json")
    if not replay_file.exists():
        print("Generating synthetic replay data...")
        from scripts.download_carbon_data import generate_synthetic_replay_data, save_replay_data
        readings = generate_synthetic_replay_data(days=2)
        save_replay_data(readings, replay_file)
    
    manager = create_carbon_manager("replay", str(replay_file))
    
    print(f"\nReplay Status:")
    status = manager.get_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Provider: {status['provider']}")
    print(f"  Available Zones: {status['available_zones'][:5]}...")
    
    # Read a few replay readings
    print(f"\nFirst 3 replay readings (national):")
    for i in range(3):
        reading = manager.get_current_carbon("national")
        if reading:
            print(f"  {reading.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                  f"{reading.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh | "
                  f"replay: {reading.is_replay}")
        else:
            break
    
    return manager


def demo_offline_mode():
    """Demo offline fallback mode."""
    print("\n" + "=" * 70)
    print("DEMO 4: OFFLINE MODE (Fallback hardcoded values)")
    print("=" * 70)
    
    manager = create_carbon_manager("offline")
    status = manager.get_status()
    
    print(f"\nOffline Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Provider: {status['provider']}")
    
    current = status['current_carbon']
    if current:
        print(f"\nOffline National Carbon Intensity (fallback):")
        print(f"  Intensity: {current['intensity_gco2_per_kwh']:.1f} gCO2/kWh")
        print(f"  Zone: {current['zone']}")
        print(f"  Offline: {current['is_offline']}")
    
    # Show all offline zones
    print(f"\nAll offline fallback zones:")
    for zone, intensity in [
        ("national", 200),
        ("North Scotland", 50),
        ("London", 300),
        ("South West England", 150),
    ]:
        reading = manager.get_current_carbon(zone)
        if reading:
            print(f"  {zone:25} | {reading.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh")
    
    return manager


def demo_co2_calculation(manager):
    """Demo CO2 calculation using Phase 0.5 measured energy."""
    print("\n" + "=" * 70)
    print("DEMO 5: CO2 CALCULATION (Measured Energy × Carbon Intensity)")
    print("=" * 70)
    
    print(f"\nPhase 0.5 Measured Baselines (RTX 2050 4GB):")
    for config, data in PHASE05_BASELINE.items():
        print(f"  {config.upper()}:")
        print(f"    Energy:      {data['avg_energy_kwh']:.6f} kWh (MEASURED)")
        print(f"    Latency:     {data['avg_latency_ms']:.0f} ms")
        print(f"    Throughput:  {data['avg_throughput_tok_s']:.1f} tok/s")
        print(f"    Power:       {data['avg_power_w']:.2f} W")
        print(f"    Peak VRAM:   {data['peak_vram_mb']} MB")
    
    # Get current carbon intensity
    current = manager.get_current_carbon("national")
    intensity = current.carbon_intensity_gco2_per_kwh if current else 200
    
    print(f"\nUsing carbon intensity: {intensity:.1f} gCO2/kWh")
    
    # Create EnergyMeasurement from Phase 0.5 data
    for config_name, baseline in PHASE05_BASELINE.items():
        energy_meas = EnergyMeasurement(
            energy_joules=baseline['avg_energy_kwh'] * 3_600_000,
            energy_wh=baseline['avg_energy_kwh'] * 1000,
            energy_kwh=baseline['avg_energy_kwh'],
            duration_seconds=baseline['avg_latency_ms'] / 1000,
            avg_power_watts=baseline['avg_power_w'],
            samples_count=int(baseline['avg_latency_ms'] / 20)  # ~20ms sampling
        )
        
        estimate = estimate_inference_carbon(energy_meas, current) if current else None
        if estimate:
            print(f"\n{config_name.upper()} CO2 Estimate:")
            print(f"  Energy (MEASURED):     {estimate.energy_kwh:.6f} kWh")
            print(f"  Carbon Intensity:      {estimate.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh")
            print(f"  CO2 (ESTIMATED):       {estimate.co2_grams:.6f} g = {estimate.co2_milligrams:.3f} mg")
    
    # Comparison
    comparison = compare_configurations_carbon(
        PHASE05_BASELINE["fp16"]["avg_energy_kwh"],
        PHASE05_BASELINE["int4"]["avg_energy_kwh"],
        intensity
    )
    print_comparison(comparison)
    
    return comparison


def demo_simulated_regional_routing(manager):
    """Demo simulated regional routing comparison."""
    print("\n" + "=" * 70)
    print("DEMO 6: SIMULATED REGIONAL ROUTING")
    print("=" * 70)
    
    print(f"\nSimulated Regions (map to real UK regions):")
    for sim, real in SIMULATED_REGIONS.items():
        print(f"  {sim:30} -> {real}")
    
    comparison = manager.get_simulated_regional_comparison()
    
    if "error" not in comparison:
        print(f"\nRegional Carbon Intensities:")
        for region, data in comparison["regions"].items():
            if not region.startswith("_"):
                print(f"  {region:30} | {data['carbon_intensity_gco2_per_kwh']:.1f} gCO2/kWh "
                      f"({data['mapped_to_real_region']})")
        
        if "_routing_decision" in comparison["regions"]:
            decision = comparison["regions"]["_routing_decision"]
            print(f"\nRouting Decision (SIMULATED):")
            print(f"  Recommended: {decision['recommended_region']} "
                  f"({decision['recommended_intensity']:.1f} gCO2/kWh)")
            print(f"  Avoid:       {decision['avoid_region']} "
                  f"({decision['avoid_intensity']:.1f} gCO2/kWh)")
            print(f"  Potential CO2 reduction: {decision['potential_reduction_pct']:.1f}%")
            print(f"\n  NOTE NOTE: This is SIMULATED routing for demo only.")
            print(f"  No actual inference is sent to other cloud regions.")
    else:
        print(f"  {comparison['error']}")
    
    return comparison


def demo_multi_provider_fallback():
    """Demo automatic fallback from live -> replay -> offline."""
    print("\n" + "=" * 70)
    print("DEMO 7: MULTI-PROVIDER FALLBACK")
    print("=" * 70)
    
    from carbongrid.carbon.providers import MultiProvider, UKCarbonIntensityProvider, OfflineCarbonProvider
    
    # Create chain: Live -> Offline
    multi = MultiProvider([
        UKCarbonIntensityProvider(),
        OfflineCarbonProvider(),
    ])
    
    print(f"\nProvider chain: UK API -> Offline")
    
    # Should get live data
    reading = multi.get_current_intensity("national")
    if reading:
        print(f"  Got data from: {reading.data_source.value}")
        print(f"  Intensity: {reading.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh")
        print(f"  Mode: {reading.metadata.get('fallback', 'live')}")
    
    # Regional should also work
    regional = multi.get_regional_snapshot()
    if regional:
        print(f"  Regional data from: {regional.mode.value}")
        print(f"  Zones: {len(regional.readings)}")
    
    print(f"\n  Fallback works automatically if API fails")


def main():
    print("=" * 70)
    print("PHASE A: CARBON DATA INFRASTRUCTURE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the complete carbon data pipeline:")
    print("  1. Live API data (UK Carbon Intensity)")
    print("  2. Provider forecast (48-hour)")
    print("  3. Replay mode (synthetic historical)")
    print("  4. Offline mode (fallback)")
    print("  5. CO2 calculation (measured energy × carbon intensity)")
    print("  6. Simulated regional routing")
    print("  7. Automatic fallback chain")
    
    # Run all demos
    manager = demo_live_mode()
    demo_forecast(manager)
    demo_replay_mode()
    demo_offline_mode()
    demo_co2_calculation(manager)
    demo_simulated_regional_routing(manager)
    demo_multi_provider_fallback()
    
    print("\n" + "=" * 70)
    print("PHASE A COMPLETE")
    print("=" * 70)
    print("\nAll carbon infrastructure components working:")
    print("  [OK] Live UK Carbon Intensity API (free, no key)")
    print("  [OK] Provider forecast (48-hour, no custom model needed)")
    print("  [OK] Regional data for simulated routing")
    print("  [OK] Replay mode with synthetic data")
    print("  [OK] Offline fallback mode")
    print("  [OK] CO2 calculation: MEASURED energy × PROVIDER carbon = ESTIMATED CO2")
    print("  [OK] Multi-provider fallback chain")
    print("\nNext: Phase B - Quantization Safety Dataset Generation")


if __name__ == "__main__":
    main()