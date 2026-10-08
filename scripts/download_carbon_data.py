#!/usr/bin/env python3
"""
Script to download historical carbon intensity data for replay/offline mode.
Fetches data from UK Carbon Intensity API and stores normalized JSON.
"""

import sys
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.carbon.providers import UKCarbonIntensityProvider
from carbongrid.carbon.models import CarbonReading, DataSource, ReadingType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_national_historical(days: int = 30) -> List[CarbonReading]:
    """
    Download national historical data.
    
    Note: UK Carbon Intensity API doesn't have a direct historical endpoint.
    We use the current + forecast endpoints and save them over time,
    or we can use the regional endpoint which has some historical data.
    
    For this script, we'll fetch current data periodically and build a dataset.
    For a complete historical dataset, we'd need to run this periodically.
    """
    provider = UKCarbonIntensityProvider()
    readings = []
    
    # Get current data
    current = provider.get_current_intensity("national")
    if current:
        readings.append(current)
        logger.info(f"Current national: {current.carbon_intensity_gco2_per_kwh} gCO2/kWh")
    
    # Get forecast (next 48 hours)
    forecast = provider.get_forecast("national", 48)
    if forecast:
        readings.extend(forecast.readings)
        logger.info(f"Forecast: {len(forecast.readings)} readings")
    
    # Get regional snapshot
    regional = provider.get_regional_snapshot()
    if regional:
        readings.extend(regional.readings.values())
        logger.info(f"Regional: {len(regional.readings)} zones")
    
    return readings


def download_regional_historical(days: int = 7) -> List[CarbonReading]:
    """
    Download regional data.
    The regional endpoint returns current + some historical periods.
    """
    provider = UKCarbonIntensityProvider()
    readings = []
    
    # Get regional data
    regional = provider.get_regional_snapshot()
    if regional:
        readings.extend(regional.readings.values())
        logger.info(f"Regional snapshot: {len(regional.readings)} zones at {regional.timestamp}")
    
    return readings


def generate_synthetic_replay_data(days: int = 7, interval_minutes: int = 30) -> List[CarbonReading]:
    """
    Generate synthetic replay data for testing when API is unavailable.
    
    Creates realistic carbon intensity patterns:
    - Daily cycles (lower at night, higher during day)
    - Regional differences
    - Some randomness
    """
    import random
    
    readings = []
    base_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Regional base intensities (gCO2/kWh) - typical UK patterns
    regional_bases = {
        "national": 200,
        "North Scotland": 40,
        "South Scotland": 80,
        "North West England": 180,
        "North East England": 160,
        "Yorkshire": 210,
        "North Wales & Mersey": 190,
        "South Wales": 240,
        "West Midlands": 270,
        "East Midlands": 250,
        "East England": 230,
        "South West England": 140,
        "South England": 220,
        "London": 290,
        "South East England": 260,
    }
    
    current = base_date
    while current < datetime.now(timezone.utc):
        # Daily cycle: lower at night (0-6), higher during day (9-17)
        hour = current.hour
        if 0 <= hour < 6:
            daily_factor = 0.7
        elif 6 <= hour < 9:
            daily_factor = 1.1
        elif 9 <= hour < 17:
            daily_factor = 1.2
        elif 17 <= hour < 22:
            daily_factor = 1.15
        else:
            daily_factor = 0.85
        
        # Weekly cycle: lower on weekends
        weekday = current.weekday()
        if weekday >= 5:  # Sat, Sun
            weekly_factor = 0.85
        else:
            weekly_factor = 1.0
        
        for zone, base in regional_bases.items():
            # Add some noise
            noise = random.uniform(0.9, 1.1)
            intensity = base * daily_factor * weekly_factor * noise
            intensity = max(10, min(500, intensity))  # Clamp
            
            reading = CarbonReading(
                timestamp=current,
                zone=zone,
                carbon_intensity_gco2_per_kwh=round(intensity, 1),
                data_source=DataSource.REPLAY,
                reading_type=ReadingType.HISTORICAL,
                is_replay=True
            )
            readings.append(reading)
        
        current += timedelta(minutes=interval_minutes)
    
    logger.info(f"Generated {len(readings)} synthetic readings over {days} days")
    return readings


def save_replay_data(readings: List[CarbonReading], filepath: Path):
    """Save readings to JSON file."""
    data = [r.to_dict() for r in readings]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(readings)} readings to {filepath}")


def load_replay_data(filepath: Path) -> List[CarbonReading]:
    """Load readings from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [CarbonReading.from_dict(item) for item in data]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download carbon intensity data for replay")
    parser.add_argument("--days", type=int, default=7, help="Days of data to generate/fetch")
    parser.add_argument("--output", type=str, default="data/replay/carbon_replay.json", help="Output file")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of API")
    parser.add_argument("--api", action="store_true", help="Try to fetch from API (limited)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if args.api:
        logger.info("Fetching from UK Carbon Intensity API...")
        readings = download_national_historical(args.days)
        readings.extend(download_regional_historical(args.days))
    else:
        logger.info("Generating synthetic replay data...")
        readings = generate_synthetic_replay_data(args.days)
    
    if readings:
        save_replay_data(readings, output_path)
        print(f"\nSuccess! {len(readings)} readings saved to {output_path}")
        
        # Show sample
        print("\nSample readings:")
        for r in readings[:5]:
            print(f"  {r.timestamp} | {r.zone:25} | {r.carbon_intensity_gco2_per_kwh:6.1f} gCO2/kWh | {r.reading_type.value}")
    else:
        print("No data collected")
        sys.exit(1)


if __name__ == "__main__":
    main()