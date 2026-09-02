import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

def generate_agricultural_dataset(num_fields: int = 84, days_history: int = 14, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a realistic multi-field agricultural observation dataset with
    temporal sensor series, weather dynamics, pest/disease progression,
    and ground-truth harvest yield calculations.
    
    Returns:
        (historical_train_df, current_prediction_df)
    """
    np.random.seed(seed)
    
    crops = ["Corn", "Soybean", "Wheat", "Cotton", "Rice"]
    varieties_by_crop = {
        "Corn": ["Pioneer P1197", "Dekalb DKC64-35", "NK Corn N78N"],
        "Soybean": ["Asgrow AG36X6", "Pioneer P31T77E", "Brevant B281EE"],
        "Wheat": ["WB9719", "SY Monument", "LCS Sonic"],
        "Cotton": ["Deltapine DP1646", "PhytoGen PHY400", "Stoneville ST4990"],
        "Rice": ["Titan Hybrid", "Diamond Long-Grain", "Jupiter Medium"]
    }
    
    stages = ["Vegetative", "Flowering", "Grain/Pod Filling", "Maturity"]
    
    start_date = datetime.now() - timedelta(days=days_history + 1)
    
    records = []
    
    for f_idx in range(1, num_fields + 1):
        field_id = f"F-{f_idx:03d}"
        crop = np.random.choice(crops)
        variety = np.random.choice(varieties_by_crop[crop])
        
        # Field inherent characteristics (soil properties)
        base_ph = np.random.uniform(5.8, 7.4)
        base_nitrogen = np.random.uniform(50, 180)
        base_phosphorus = np.random.uniform(20, 70)
        base_potassium = np.random.uniform(70, 220)
        
        # Determine scenario for this field
        # 15% drought stress, 12% fungal disease, 8% pest attack, 10% nutrient deficit, 55% healthy/moderate
        scenario_rand = np.random.rand()
        if scenario_rand < 0.15:
            scenario = "drought"
        elif scenario_rand < 0.27:
            scenario = "disease"
        elif scenario_rand < 0.35:
            scenario = "pest"
        elif scenario_rand < 0.45:
            scenario = "nutrient_deficit"
        else:
            scenario = "healthy"
            
        initial_age = int(np.random.uniform(30, 85))
        stage_idx = min(3, initial_age // 25)
        
        # Initial sensor states
        cur_moisture = 55.0 if scenario != "drought" else 32.0
        cur_disease = 5.0 if scenario != "disease" else 28.0
        cur_pest = int(np.random.uniform(5, 20)) if scenario != "pest" else 65
        cur_ndvi = 0.72
        
        for d in range(days_history + 1):
            obs_date = (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
            age = initial_age + d
            stage = stages[min(3, age // 25)]
            
            # Weather variation with day-to-day correlation
            day_temp = np.random.normal(26.0, 3.5)
            day_humidity = np.random.normal(65.0, 10.0)
            day_rain = 0.0
            
            if np.random.rand() < 0.2:
                day_rain = np.random.exponential(12.0)
                day_humidity = min(98.0, day_humidity + 20.0)
                day_temp -= 3.0
                
            day_wind = np.clip(np.random.normal(14.0, 5.0), 2.0, 45.0)
            day_solar = np.clip(np.random.normal(22.0, 4.0) - (day_rain * 0.4), 8.0, 32.0)
            
            # Irrigation & Fertilizer decisions
            irrig = 0.0
            fert = 0.0
            if scenario != "drought" and cur_moisture < 40 and np.random.rand() < 0.6:
                irrig = np.random.uniform(15.0, 35.0)
            if age in [40, 65] and scenario != "nutrient_deficit":
                fert = np.random.uniform(40.0, 90.0)
                
            # Dynamic updates
            cur_moisture += (day_rain * 0.8) + (irrig * 0.9) - (day_temp * 0.25) + np.random.normal(0, 1.5)
            if scenario == "drought":
                cur_moisture -= 1.2
            cur_moisture = float(np.clip(cur_moisture, 8.0, 85.0))
            
            if scenario == "disease":
                cur_disease += np.random.uniform(1.5, 4.0) if day_humidity > 65 else np.random.uniform(0.2, 1.2)
            else:
                cur_disease = max(2.0, cur_disease + np.random.normal(0, 0.5))
            cur_disease = float(np.clip(cur_disease, 0.0, 95.0))
            
            if scenario == "pest":
                cur_pest = int(np.clip(cur_pest + np.random.randint(-3, 12), 10, 220))
            else:
                cur_pest = int(np.clip(cur_pest + np.random.randint(-4, 4), 2, 45))
                
            # NDVI degradation with stress
            ndvi_loss = (cur_disease * 0.002) + (max(0, 35 - cur_moisture) * 0.004) + (cur_pest * 0.001)
            cur_ndvi = float(np.clip(0.80 - ndvi_loss + np.random.normal(0, 0.02), 0.15, 0.92))
            
            # Soil temp & pH
            soil_t = float(np.clip(day_temp * 0.85 + 2.0 + np.random.normal(0, 0.8), 10.0, 42.0))
            soil_ph = float(np.clip(base_ph + np.random.normal(0, 0.05), 4.5, 8.8))
            
            # Nutrients
            n_val = float(np.clip(base_nitrogen + fert - (age * 0.4), 15.0, 350.0))
            p_val = float(np.clip(base_phosphorus + (fert * 0.3) - (age * 0.1), 8.0, 150.0))
            k_val = float(np.clip(base_potassium + (fert * 0.2) - (age * 0.15), 25.0, 300.0))
            
            records.append({
                "field_id": field_id,
                "date": obs_date,
                "crop_type": crop,
                "variety": variety,
                "crop_age_days": age,
                "growth_stage": stage,
                "soil_moisture": round(cur_moisture, 1),
                "soil_temperature": round(soil_t, 1),
                "soil_ph": round(soil_ph, 2),
                "nitrogen": round(n_val, 1),
                "phosphorus": round(p_val, 1),
                "potassium": round(k_val, 1),
                "temperature": round(day_temp, 1),
                "humidity": round(day_humidity, 1),
                "rainfall": round(day_rain, 1),
                "wind_speed": round(day_wind, 1),
                "solar_radiation": round(day_solar, 1),
                "irrigation_amount": round(irrig, 1),
                "fertilizer_amount": round(fert, 1),
                "disease_score": round(cur_disease, 1),
                "pest_count": int(cur_pest),
                "ndvi": round(cur_ndvi, 3),
                "scenario": scenario
            })
            
    df_all = pd.DataFrame(records)
    
    # Calculate realistic actual_yield for training records based on agronomic production function
    from backend.ml.feature_engineering import CROP_POTENTIAL_YIELDS
    
    def calculate_yield(row):
        pot = CROP_POTENTIAL_YIELDS.get(row["crop_type"], 5.0)
        # Penalties
        drought_pen = max(0, (38.0 - row["soil_moisture"]) / 38.0) * 0.45
        disease_pen = (row["disease_score"] / 100.0) * 0.40
        pest_pen = min(0.35, (row["pest_count"] / 180.0) * 0.35)
        heat_pen = max(0, (row["temperature"] - 31.0) / 12.0) * 0.25
        fert_pen = max(0, (70.0 - row["nitrogen"]) / 70.0) * 0.20
        ndvi_bonus = (row["ndvi"] - 0.5) * 0.3
        
        total_loss_fraction = np.clip(drought_pen + disease_pen + pest_pen + heat_pen + fert_pen - ndvi_bonus, 0.02, 0.85)
        # Add slight natural field variation
        actual = pot * (1.0 - total_loss_fraction) * np.random.normal(1.0, 0.04)
        return round(float(np.clip(actual, pot * 0.12, pot * 1.08)), 2)

    df_all["actual_yield"] = df_all.apply(calculate_yield, axis=1)
    
    # Training set has all history with actual_yield
    historical_train_df = df_all.drop(columns=["scenario"])
    
    # Prediction set: Latest snapshot for each field, WITHOUT actual_yield column (prevent leakage)
    latest_dates = df_all.groupby("field_id")["date"].max().reset_index()
    prediction_df = pd.merge(df_all, latest_dates, on=["field_id", "date"]).drop(columns=["actual_yield", "scenario"])
    
    return historical_train_df, prediction_df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    train_df, pred_df = generate_agricultural_dataset(num_fields=84, days_history=14)
    train_df.to_csv("data/historical_training_data.csv", index=False)
    pred_df.to_csv("data/sample_agricultural_data.csv", index=False)
    print(f"Generated {len(train_df)} training records and {len(pred_df)} prediction fields.")
