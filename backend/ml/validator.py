import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Standard schema definition
REQUIRED_COLUMNS = [
    "field_id", "date", "crop_type", "variety", "crop_age_days", "growth_stage",
    "soil_moisture", "soil_temperature", "soil_ph", "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "rainfall", "wind_speed", "solar_radiation",
    "irrigation_amount", "fertilizer_amount", "disease_score", "pest_count", "ndvi"
]

OPTIONAL_TRAIN_COLUMNS = ["actual_yield"]

# Physical valid ranges for agronomic sensor values
VALIDATION_RANGES = {
    "crop_age_days": (1, 300),
    "soil_moisture": (0.0, 100.0),      # percentage
    "soil_temperature": (-10.0, 60.0),  # deg C
    "soil_ph": (3.5, 9.5),
    "nitrogen": (0.0, 400.0),           # kg/ha or mg/kg
    "phosphorus": (0.0, 200.0),         # kg/ha
    "potassium": (0.0, 400.0),          # kg/ha
    "temperature": (-15.0, 55.0),       # deg C
    "humidity": (0.0, 100.0),           # %
    "rainfall": (0.0, 300.0),           # mm/day
    "wind_speed": (0.0, 150.0),         # km/h
    "solar_radiation": (0.0, 40.0),     # MJ/m^2/day
    "irrigation_amount": (0.0, 200.0),  # mm
    "fertilizer_amount": (0.0, 500.0),  # kg/ha
    "disease_score": (0.0, 100.0),      # index 0-100
    "pest_count": (0, 500),             # count/sqm
    "ndvi": (-0.2, 1.0)                 # vegetation index
}

# Alias dictionary for smart auto-column mapping
COLUMN_ALIASES = {
    "field_id": ["field_id", "field", "fieldid", "plot_id", "zone_id", "farm_id"],
    "date": ["date", "timestamp", "observation_date", "time", "record_date"],
    "crop_type": ["crop_type", "crop", "crop_name", "plant_type"],
    "variety": ["variety", "cultivar", "crop_variety", "hybrid"],
    "crop_age_days": ["crop_age_days", "age", "days_after_planting", "dap", "crop_age"],
    "growth_stage": ["growth_stage", "stage", "phenological_stage", "phase"],
    "soil_moisture": ["soil_moisture", "moisture", "sm", "soil_water_content", "vwc", "soil_moisture_pct"],
    "soil_temperature": ["soil_temperature", "soil_temp", "soil_t", "ground_temp"],
    "soil_ph": ["soil_ph", "ph", "soilph"],
    "nitrogen": ["nitrogen", "n_level", "soil_n", "n", "nitrogen_kg_ha"],
    "phosphorus": ["phosphorus", "p_level", "soil_p", "p", "phosphorus_kg_ha"],
    "potassium": ["potassium", "k_level", "soil_k", "k", "potassium_kg_ha"],
    "temperature": ["temperature", "temp", "air_temperature", "air_temp", "t_avg"],
    "humidity": ["humidity", "rel_humidity", "rh", "relative_humidity"],
    "rainfall": ["rainfall", "rain", "precipitation", "precip", "rain_mm"],
    "wind_speed": ["wind_speed", "wind", "wind_velocity", "wind_kmh"],
    "solar_radiation": ["solar_radiation", "radiation", "solar_rad", "sunlight", "rad"],
    "irrigation_amount": ["irrigation_amount", "irrigation", "water_applied", "irrig_mm"],
    "fertilizer_amount": ["fertilizer_amount", "fertilizer", "fert_applied", "fert_kg_ha"],
    "disease_score": ["disease_score", "disease_index", "disease_severity", "disease_pct", "infection_rate"],
    "pest_count": ["pest_count", "pest_density", "pests", "insects_count"],
    "ndvi": ["ndvi", "veg_index", "normalized_diff_veg_idx", "ndvi_score"],
    "actual_yield": ["actual_yield", "yield", "final_yield", "harvest_yield", "yield_tons_acre"]
}

def auto_detect_column_mapping(df_columns: List[str]) -> Dict[str, Optional[str]]:
    """Suggests mapping from standard schema names to DataFrame column names."""
    mapping = {}
    lower_df_cols = {col.lower().strip().replace(" ", "_").replace("-", "_"): col for col in df_columns}
    
    for std_col, aliases in COLUMN_ALIASES.items():
        matched = None
        for alias in aliases:
            norm_alias = alias.lower().replace(" ", "_").replace("-", "_")
            if norm_alias in lower_df_cols:
                matched = lower_df_cols[norm_alias]
                break
        mapping[std_col] = matched
    return mapping

def validate_dataframe(df: pd.DataFrame, is_training: bool = False) -> Dict[str, Any]:
    """
    Performs comprehensive data validation:
    - Missing required columns
    - Missing value percentages (missingness)
    - Duplicate detection
    - Range bound violations
    - Data leakage warnings (e.g. actual_yield provided in inference mode)
    """
    report = {
        "valid": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_columns": [],
        "missingness": {},
        "duplicates_count": 0,
        "out_of_range_warnings": [],
        "data_leakage_warnings": [],
        "summary": "Data validated successfully."
    }
    
    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            report["missing_columns"].append(col)
    
    if is_training and "actual_yield" not in df.columns:
        report["missing_columns"].append("actual_yield (required for training)")
    
    # Check data leakage in inference mode
    if not is_training and "actual_yield" in df.columns:
        report["data_leakage_warnings"].append(
            "Column 'actual_yield' was found in prediction dataset. It will be excluded from features to prevent data leakage."
        )
    
    if report["missing_columns"]:
        report["valid"] = False
        report["summary"] = f"Missing {len(report['missing_columns'])} required columns: {', '.join(report['missing_columns'][:4])}"
        return report

    # Missing value analysis
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            pct = round((null_count / len(df)) * 100, 2)
            report["missingness"][col] = {"count": null_count, "pct": pct}

    # Duplicates check (by field_id and date if present)
    if "field_id" in df.columns and "date" in df.columns:
        dups = df.duplicated(subset=["field_id", "date"]).sum()
        report["duplicates_count"] = int(dups)
    else:
        report["duplicates_count"] = int(df.duplicated().sum())

    # Range validation
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            below = (df[col] < min_val).sum()
            above = (df[col] > max_val).sum()
            if below > 0 or above > 0:
                report["out_of_range_warnings"].append({
                    "column": col,
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                    "below_min_count": int(below),
                    "above_max_count": int(above),
                    "observed_min": float(df[col].min()),
                    "observed_max": float(df[col].max())
                })

    return report

# Default fallback values for standard agronomic columns
COLUMN_DEFAULTS = {
    "field_id": "F-001",
    "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "crop_type": "Corn",
    "variety": "Standard Hybrid",
    "crop_age_days": 55,
    "growth_stage": "Flowering",
    "soil_moisture": 45.0,
    "soil_temperature": 22.0,
    "soil_ph": 6.5,
    "nitrogen": 110.0,
    "phosphorus": 45.0,
    "potassium": 130.0,
    "temperature": 25.0,
    "humidity": 65.0,
    "rainfall": 5.0,
    "wind_speed": 12.0,
    "solar_radiation": 22.0,
    "irrigation_amount": 0.0,
    "fertilizer_amount": 0.0,
    "disease_score": 10.0,
    "pest_count": 15,
    "ndvi": 0.72
}

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans types, clips physical ranges, imputes missing values and ensures all required features exist."""
    cleaned = df.copy()
    
    # Ensure all required standard columns exist
    for col, default_val in COLUMN_DEFAULTS.items():
        if col not in cleaned.columns:
            cleaned[col] = default_val
    
    # Date formatting
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["date"] = cleaned["date"].fillna(pd.Timestamp.now())
    
    # Clip numeric ranges to sensible physical bounds
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].clip(lower=min_val, upper=max_val)
            # Impute missing with median or field median
            if cleaned[col].isnull().any():
                med = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(med if not pd.isna(med) else COLUMN_DEFAULTS.get(col, (min_val + max_val)/2.0))
    
    # Categoricals
    cat_cols = ["crop_type", "variety", "growth_stage"]
    for cat in cat_cols:
        if cat in cleaned.columns:
            cleaned[cat] = cleaned[cat].fillna("Unknown").astype(str)
            
    if "field_id" in cleaned.columns:
        cleaned["field_id"] = cleaned["field_id"].astype(str)
        # Ensure proper field_id formatting
        cleaned["field_id"] = cleaned["field_id"].apply(lambda x: x if x.startswith("F-") else f"F-{x}")

    return cleaned
