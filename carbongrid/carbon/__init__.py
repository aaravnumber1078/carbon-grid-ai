"""
CarbonGrid Carbon Module

Provides carbon intensity data, forecasting, and CO2 estimation for AI inference.
"""

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
)

from carbongrid.carbon.providers import (
    CarbonDataProvider,
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

__all__ = [
    # Models
    "CarbonReading",
    "CarbonForecast", 
    "RegionalCarbonSnapshot",
    "DataSource",
    "DataMode",
    "ReadingType",
    "SIMULATED_REGIONS",
    "UK_REGIONS",
    "calculate_co2_grams",
    "calculate_co2_mg",
    "calculate_co2_kg",
    # Providers
    "CarbonDataProvider",
    "UKCarbonIntensityProvider",
    "ReplayCarbonProvider",
    "OfflineCarbonProvider",
    "MultiProvider",
    # Calculator
    "EnergyMeasurement",
    "CarbonEstimate",
    "estimate_inference_carbon",
    "compare_configurations_carbon",
    "PHASE05_BASELINE",
    # Manager
    "CarbonManager",
    "CarbonMode",
    "create_carbon_manager",
]