import pandas as pd
import numpy as np
from typing import List, Tuple

# Crop baseline potentials (tons/acre) by crop type
CROP_POTENTIAL_YIELDS = {
    "Corn": 7.5,
    "Soybean": 3.8,
    "Wheat": 4.5,
    "Cotton": 2.2,
    "Rice": 6.0,
    "Tomato": 25.0,
    "Potato": 18.0,
    "Canola": 2.8
}

DEFAULT_POTENTIAL = 5.0

# Expected growth cycle duration (days)
CROP_CYCLE_DAYS = {
    "Corn": 120,
    "Soybean": 110,
    "Wheat": 140,
    "Cotton": 160,
    "Rice": 130,
    "Tomato": 90,
    "Potato": 105,
    "Canola": 100
}

def compute_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based, stage-aware, interaction, and stress index features
    while preserving temporal integrity and preventing data leakage.
    """
    df_feat = df.copy()
    
    # Ensure sorting for rolling operations if date exists
    if "date" in df_feat.columns and "field_id" in df_feat.columns:
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        df_feat = df_feat.sort_values(by=["field_id", "date"]).reset_index(drop=True)
    
    # Crop Potential Yield Baseline
    df_feat["expected_yield_baseline"] = df_feat["crop_type"].map(CROP_POTENTIAL_YIELDS).fillna(DEFAULT_POTENTIAL)
    
    # Days to estimated harvest
    total_cycle = df_feat["crop_type"].map(CROP_CYCLE_DAYS).fillna(120)
    df_feat["days_to_harvest"] = (total_cycle - df_feat["crop_age_days"]).clip(lower=0)
    df_feat["growth_progress_pct"] = (df_feat["crop_age_days"] / total_cycle).clip(lower=0.0, upper=1.2)
    
    # Grouped rolling & trend features per field
    if "field_id" in df_feat.columns and len(df_feat["field_id"].unique()) < len(df_feat):
        # 3-day and 7-day rolling statistics
        for window in [3, 7]:
            df_feat[f"soil_moisture_roll_{window}d_avg"] = df_feat.groupby("field_id")["soil_moisture"].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df_feat[f"temp_roll_{window}d_avg"] = df_feat.groupby("field_id")["temperature"].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df_feat[f"humidity_roll_{window}d_avg"] = df_feat.groupby("field_id")["humidity"].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df_feat[f"rainfall_roll_{window}d_sum"] = df_feat.groupby("field_id")["rainfall"].transform(
                lambda s: s.rolling(window, min_periods=1).sum()
            )
            
        # Trends (rate of change over recent records)
        df_feat["moisture_trend_3d"] = df_feat.groupby("field_id")["soil_moisture"].transform(
            lambda s: s.diff(periods=2).fillna(0)
        )
        df_feat["disease_growth_rate_3d"] = df_feat.groupby("field_id")["disease_score"].transform(
            lambda s: s.diff(periods=2).fillna(0)
        )
        df_feat["ndvi_decline_velocity"] = df_feat.groupby("field_id")["ndvi"].transform(
            lambda s: (s.iloc[0] - s).fillna(0) if len(s) > 0 else 0
        )
    else:
        # Fallback if dataset is a snapshot of 1 record per field
        df_feat["soil_moisture_roll_3d_avg"] = df_feat["soil_moisture"]
        df_feat["soil_moisture_roll_7d_avg"] = df_feat["soil_moisture"]
        df_feat["temp_roll_3d_avg"] = df_feat["temperature"]
        df_feat["temp_roll_7d_avg"] = df_feat["temperature"]
        df_feat["humidity_roll_3d_avg"] = df_feat["humidity"]
        df_feat["humidity_roll_7d_avg"] = df_feat["humidity"]
        df_feat["rainfall_roll_3d_sum"] = df_feat["rainfall"] * 2.5
        df_feat["rainfall_roll_7d_sum"] = df_feat["rainfall"] * 6.0
        df_feat["moisture_trend_3d"] = 0.0
        df_feat["disease_growth_rate_3d"] = 0.0
        df_feat["ndvi_decline_velocity"] = 0.0

    # Interaction Features
    # 1. Heat & Drought Stress: High Temp + Low Soil Moisture
    # High temp > 30C, Low moisture < 30%
    heat_term = np.maximum(0, (df_feat["temperature"] - 25.0) / 15.0)
    drought_term = np.maximum(0, (40.0 - df_feat["soil_moisture"]) / 40.0)
    df_feat["heat_drought_interaction"] = heat_term * drought_term

    # 2. Fungal Disease Risk: High Disease Score + High Humidity + Warm Temp
    humid_term = np.maximum(0, (df_feat["humidity"] - 65.0) / 35.0)
    temp_fungal_term = np.clip((df_feat["temperature"] - 15.0) / 15.0, 0, 1.2)
    df_feat["disease_humidity_interaction"] = (df_feat["disease_score"] / 100.0) * humid_term * temp_fungal_term

    # 3. Water Stress Index (0 = optimal, 1 = severe stress)
    # Optimal soil moisture is roughly 45% - 70%
    moisture_deficit = np.maximum(0, (45.0 - df_feat["soil_moisture"]) / 45.0)
    moisture_excess = np.maximum(0, (df_feat["soil_moisture"] - 80.0) / 20.0)
    df_feat["water_stress_index"] = np.clip(moisture_deficit + 0.5 * moisture_excess, 0.0, 1.0)

    # 4. Nutrition Balance & Stress Index
    # Ideal N:P:K roughly 4:2:1 or balanced proportion
    total_npk = df_feat["nitrogen"] + df_feat["phosphorus"] + df_feat["potassium"] + 1e-5
    df_feat["n_ratio"] = df_feat["nitrogen"] / total_npk
    df_feat["p_ratio"] = df_feat["phosphorus"] / total_npk
    df_feat["k_ratio"] = df_feat["potassium"] / total_npk
    
    # Nutrition deficit penalty
    n_stress = np.maximum(0, (80.0 - df_feat["nitrogen"]) / 80.0)
    p_stress = np.maximum(0, (30.0 - df_feat["phosphorus"]) / 30.0)
    k_stress = np.maximum(0, (60.0 - df_feat["potassium"]) / 60.0)
    df_feat["nutrition_stress_index"] = np.clip((0.5 * n_stress + 0.3 * p_stress + 0.2 * k_stress), 0.0, 1.0)

    # 5. Pest Pressure Index
    df_feat["pest_pressure_index"] = np.clip(df_feat["pest_count"] / 120.0, 0.0, 1.5)

    # 6. Overall Vegetative Vigor Deficit
    # Expected NDVI at peak stage is ~0.75-0.85
    df_feat["ndvi_vigor_deficit"] = np.maximum(0, 0.75 - df_feat["ndvi"])

    return df_feat

# Feature columns used for ML models
MODEL_FEATURE_COLS = [
    "crop_age_days", "growth_progress_pct", "days_to_harvest",
    "soil_moisture", "soil_temperature", "soil_ph",
    "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "rainfall", "wind_speed", "solar_radiation",
    "irrigation_amount", "fertilizer_amount", "disease_score", "pest_count", "ndvi",
    "soil_moisture_roll_3d_avg", "soil_moisture_roll_7d_avg",
    "temp_roll_3d_avg", "temp_roll_7d_avg",
    "humidity_roll_3d_avg", "humidity_roll_7d_avg",
    "rainfall_roll_3d_sum", "rainfall_roll_7d_sum",
    "moisture_trend_3d", "disease_growth_rate_3d", "ndvi_decline_velocity",
    "heat_drought_interaction", "disease_humidity_interaction",
    "water_stress_index", "nutrition_stress_index", "pest_pressure_index",
    "ndvi_vigor_deficit", "n_ratio", "p_ratio", "k_ratio"
]

CATEGORICAL_COLS = ["crop_type", "variety", "growth_stage"]
