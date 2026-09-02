import os
import io
import uuid
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ml.validator import validate_dataframe, auto_detect_column_mapping, clean_and_impute_data, REQUIRED_COLUMNS
from backend.ml.dataset_generator import generate_agricultural_dataset
from backend.ml.models import engine

app = FastAPI(
    title="CropNow - AI Crop Failure & Yield-Loss Early Warning System",
    description="End-to-end Agronomic Decision Support & Predictive AI System",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for datasets & prediction runs
DATASETS_STORE: Dict[str, pd.DataFrame] = {}
VALIDATION_STORE: Dict[str, Dict[str, Any]] = {}
PREDICTIONS_STORE: Dict[str, Dict[str, Any]] = {}

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Pydantic Schemas
class ColumnMappingRequest(BaseModel):
    dataset_id: str
    mapping: Dict[str, str]

class SimulationRequest(BaseModel):
    field_id: str
    irrigation_adjustment_mm: float = 0.0
    fertilizer_adjustment_kg: float = 0.0
    fungicide_applied: bool = False
    temperature_anomaly: float = 0.0

from backend.ml.pdf_extractor import extract_dataframe_from_pdf, create_sample_pdf_report

@app.on_event("startup")
def startup_event():
    """Generates initial agricultural data and pre-trains model on startup."""
    train_csv_path = os.path.join(DATA_DIR, "historical_training_data.csv")
    sample_csv_path = os.path.join(DATA_DIR, "sample_agricultural_data.csv")
    sample_pdf_path = os.path.join(DATA_DIR, "sample_field_observations.pdf")
    
    train_df, sample_df = generate_agricultural_dataset(num_fields=84, days_history=14, seed=42)
    train_df.to_csv(train_csv_path, index=False)
    sample_df.to_csv(sample_csv_path, index=False)
    
    try:
        create_sample_pdf_report(sample_pdf_path, sample_df)
    except Exception as e:
        print("PDF report generation notice:", e)
    
    # Train the ML models
    engine.train_and_evaluate(train_df)
    
    # Pre-run sample dataset so dashboard starts loaded
    demo_id = "demo-84-fields"
    DATASETS_STORE[demo_id] = sample_df
    VALIDATION_STORE[demo_id] = validate_dataframe(sample_df)
    results_df, portfolio_summary = engine.predict_field_observations(sample_df)
    
    PREDICTIONS_STORE[demo_id] = {
        "results": results_df.to_dict(orient="records"),
        "portfolio_summary": portfolio_summary,
        "dataset_id": demo_id,
        "model_version": engine.version
    }

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handles Multi-File Batch and Single File uploads (CSV, PDF, Excel).
    Extracts, normalizes, and merges all uploaded field observations into a unified dataset.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
        
    extracted_dfs = []
    file_summaries = []
    
    for file in files:
        filename = file.filename.lower()
        content = await file.read()
        try:
            if filename.endswith(".pdf"):
                raw_df = extract_dataframe_from_pdf(content)
                fmt = "PDF"
            elif filename.endswith(".csv"):
                raw_df = pd.read_csv(io.BytesIO(content))
                fmt = "CSV"
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                raw_df = pd.read_excel(io.BytesIO(content))
                fmt = "Excel"
            else:
                continue
                
            if not raw_df.empty:
                # Apply column auto-mapping on each file first
                mapping = auto_detect_column_mapping(list(raw_df.columns))
                rename_dict = {v: k for k, v in mapping.items() if v and v in raw_df.columns}
                mapped_df = raw_df.rename(columns=rename_dict)
                cleaned_df = clean_and_impute_data(mapped_df)
                extracted_dfs.append(cleaned_df)
                file_summaries.append(f"{file.filename} ({len(cleaned_df)} rows, {fmt})")
        except Exception as e:
            print(f"Error parsing {file.filename}:", e)

    if not extracted_dfs:
        raise HTTPException(status_code=400, detail="Could not extract observations from any of the provided files.")
        
    # Merge and concatenate all extracted dataframes
    merged_df = pd.concat(extracted_dfs, ignore_index=True)
    # Deduplicate if duplicate field_id + date exist
    if "field_id" in merged_df.columns and "date" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["field_id", "date"], keep="last")
    else:
        merged_df = merged_df.drop_duplicates(keep="last")
        
    dataset_id = str(uuid.uuid4())
    DATASETS_STORE[dataset_id] = merged_df
    
    mapping_suggestion = auto_detect_column_mapping(list(merged_df.columns))
    validation_report = validate_dataframe(merged_df)
    VALIDATION_STORE[dataset_id] = validation_report
    
    display_name = ", ".join([f.filename for f in files]) if len(files) <= 2 else f"{len(files)} files merged ({files[0].filename}, {files[1].filename}...)"
    
    return {
        "dataset_id": dataset_id,
        "filename": display_name,
        "format": f"Multi-File Batch ({len(files)} files)" if len(files) > 1 else ("PDF" if files[0].filename.lower().endswith(".pdf") else "CSV/Excel"),
        "files_count": len(files),
        "file_summaries": file_summaries,
        "total_rows": len(merged_df),
        "columns": list(merged_df.columns),
        "column_mapping_suggestion": mapping_suggestion,
        "validation_report": validation_report
    }

@app.post("/api/map-columns")
async def map_columns(req: ColumnMappingRequest):
    """Applies user-defined or auto-detected column renaming."""
    if req.dataset_id not in DATASETS_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    df = DATASETS_STORE[req.dataset_id].copy()
    rename_dict = {v: k for k, v in req.mapping.items() if v and v in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Clean & impute
    cleaned_df = clean_and_impute_data(df)
    DATASETS_STORE[req.dataset_id] = cleaned_df
    
    validation_report = validate_dataframe(cleaned_df)
    VALIDATION_STORE[req.dataset_id] = validation_report
    
    return {
        "dataset_id": req.dataset_id,
        "validation_report": validation_report,
        "cleaned_columns": list(cleaned_df.columns)
    }

@app.get("/api/dataset/{dataset_id}/validation")
async def get_validation_report(dataset_id: str):
    """Returns detailed data validation diagnostics."""
    if dataset_id not in VALIDATION_STORE:
        raise HTTPException(status_code=404, detail="Validation report not found.")
    return VALIDATION_STORE[dataset_id]

@app.get("/api/dataset/{dataset_id}/data")
async def get_dataset_raw_data(dataset_id: str, limit: int = 500):
    """Returns raw observation records and sensor summary statistics."""
    if dataset_id not in DATASETS_STORE:
        # Fallback to demo dataset if available
        if "demo-84-fields" in DATASETS_STORE:
            dataset_id = "demo-84-fields"
        else:
            raise HTTPException(status_code=404, detail="Dataset not found.")
            
    df = DATASETS_STORE[dataset_id]
    
    # Calculate quick numeric stats
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2)
        }
        
    records = df.head(limit).fillna("").to_dict(orient="records")
    
    return {
        "dataset_id": dataset_id,
        "total_records": len(df),
        "columns": list(df.columns),
        "stats": stats,
        "data": records
    }

@app.post("/api/predict")
async def run_prediction(dataset_id: str = Form("demo-84-fields")):
    """Executes full ML prediction pipeline on specified dataset."""
    if dataset_id not in DATASETS_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    df = DATASETS_STORE[dataset_id]
    cleaned_df = clean_and_impute_data(df)
    
    results_df, portfolio_summary = engine.predict_field_observations(cleaned_df)
    
    prediction_id = dataset_id if dataset_id.startswith("demo") else str(uuid.uuid4())
    PREDICTIONS_STORE[prediction_id] = {
        "results": results_df.to_dict(orient="records"),
        "portfolio_summary": portfolio_summary,
        "dataset_id": dataset_id,
        "model_version": engine.version
    }
    
    return {
        "prediction_id": prediction_id,
        "portfolio_summary": portfolio_summary,
        "total_fields_analyzed": len(results_df),
        "results_sample": results_df.head(10).to_dict(orient="records")
    }

@app.get("/api/predictions/{prediction_id}")
async def get_predictions(
    prediction_id: str,
    risk_level: Optional[str] = None,
    crop_type: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: Optional[str] = "priority_rank",
    sort_asc: bool = True
):
    """Returns full prediction results with filtering, sorting, and pagination."""
    if prediction_id not in PREDICTIONS_STORE:
        raise HTTPException(status_code=404, detail="Prediction results not found.")
        
    data = PREDICTIONS_STORE[prediction_id]
    results = data["results"]
    
    # Filter
    filtered = results
    if risk_level and risk_level.upper() != "ALL":
        filtered = [r for r in filtered if r["risk_level"].upper() == risk_level.upper()]
    if crop_type and crop_type.upper() != "ALL":
        filtered = [r for r in filtered if r["crop_type"].upper() == crop_type.upper()]
    if search:
        s = search.lower()
        filtered = [
            r for r in filtered if s in r["field_id"].lower() or s in r["crop_type"].lower() or s in r["variety"].lower() or s in r["primary_risk_factor"].lower()
        ]
        
    # Sort
    if sort_by:
        try:
            filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=not sort_asc)
        except Exception:
            pass
            
    return {
        "prediction_id": prediction_id,
        "portfolio_summary": data["portfolio_summary"],
        "model_version": data["model_version"],
        "total_records": len(filtered),
        "results": filtered
    }

@app.get("/api/field/{prediction_id}/{field_id}")
async def get_field_deepdive(prediction_id: str, field_id: str):
    """Returns rich single-field deep dive with SHAP attribution, trajectory, and sensor diagnostics."""
    if prediction_id not in PREDICTIONS_STORE:
        raise HTTPException(status_code=404, detail="Prediction results not found.")
        
    records = PREDICTIONS_STORE[prediction_id]["results"]
    field_record = next((r for r in records if r["field_id"].lower() == field_id.lower()), None)
    
    if not field_record:
        raise HTTPException(status_code=404, detail=f"Field {field_id} not found.")
        
    return field_record

@app.post("/api/simulate")
async def run_simulation(sim: SimulationRequest, prediction_id: str = "demo-84-fields"):
    """
    What-If Counterfactual Sandbox:
    Allows agronomists to simulate the effect of interventions (irrigation, fertilizer, fungicide)
    and observe real-time yield recovery and risk mitigation.
    """
    if prediction_id not in PREDICTIONS_STORE:
        prediction_id = "demo-84-fields"
        
    records = PREDICTIONS_STORE.get(prediction_id, {}).get("results", [])
    if not records:
        raise HTTPException(status_code=404, detail="No prediction records available.")
        
    orig = next((r for r in records if r["field_id"].lower() == sim.field_id.lower()), None)
    if not orig:
        orig = records[0]
        
    # Create counterfactual row
    sensors = dict(orig.get("sensors", {}))
    
    defaults = {
        "soil_moisture": 50.0, "soil_temperature": 22.0, "soil_ph": 6.5,
        "nitrogen": 100.0, "phosphorus": 40.0, "potassium": 120.0,
        "temperature": 25.0, "humidity": 65.0, "rainfall": 5.0,
        "wind_speed": 12.0, "solar_radiation": 22.0, "irrigation_amount": 0.0,
        "fertilizer_amount": 0.0, "disease_score": 10.0, "pest_count": 15, "ndvi": 0.75
    }
    for k, v in defaults.items():
        if k not in sensors:
            sensors[k] = v
            
    # Apply simulated interventions
    import numpy as np
    sensors["soil_moisture"] = float(np.clip(sensors["soil_moisture"] + (sim.irrigation_adjustment_mm * 0.8), 5.0, 95.0))
    sensors["nitrogen"] = float(np.clip(sensors["nitrogen"] + (sim.fertilizer_adjustment_kg * 0.6), 10.0, 400.0))
    sensors["fertilizer_amount"] = float(sensors["fertilizer_amount"] + sim.fertilizer_adjustment_kg)
    sensors["irrigation_amount"] = float(sensors["irrigation_amount"] + sim.irrigation_adjustment_mm)
    if sim.fungicide_applied:
        sensors["disease_score"] = float(max(2.0, sensors["disease_score"] * 0.25))
    sensors["temperature"] = float(sensors["temperature"] + sim.temperature_anomaly)
    
    sim_row = {
        "field_id": orig["field_id"],
        "date": orig["prediction_date"],
        "crop_type": orig["crop_type"],
        "variety": orig["variety"],
        "growth_stage": orig["growth_stage"],
        "crop_age_days": orig["crop_age_days"],
        **sensors
    }
    
    sim_df = pd.DataFrame([sim_row])
    sim_res_df, _ = engine.predict_field_observations(sim_df)
    sim_res = sim_res_df.iloc[0].to_dict()
    
    yield_delta = round(sim_res["predicted_yield"] - orig["predicted_yield"], 2)
    risk_prob_delta = round(sim_res["failure_probability"] - orig["failure_probability"], 2)
    loss_reduction_pct = round(orig["yield_loss_percentage"] - sim_res["yield_loss_percentage"], 1)
    
    # Crop commodity prices per ton in Indian Rupees (INR)
    crop_prices_inr = {"Corn": 22000, "Soybean": 44000, "Wheat": 23000, "Cotton": 65000, "Rice": 32000}
    price_per_ton_inr = crop_prices_inr.get(orig["crop_type"], 28000)
    
    revenue_saved_per_acre = round(max(0.0, yield_delta * price_per_ton_inr), 0)
    treatment_cost_per_acre = round(
        (sim.irrigation_adjustment_mm * 85.0) +
        (sim.fertilizer_adjustment_kg * 48.0) +
        (2400.0 if sim.fungicide_applied else 0.0), 0
    )
    net_benefit_per_acre = round(revenue_saved_per_acre - treatment_cost_per_acre, 0)
    roi_multiple = round(revenue_saved_per_acre / max(1.0, treatment_cost_per_acre), 1) if treatment_cost_per_acre > 0 else 0.0

    # Realistic progressive counterfactual trajectory as interventions take effect across Days 1 -> 3 -> 5 -> 7
    today_risk = round(orig["failure_probability"] * 100)
    final_sim_risk = round(sim_res["failure_probability"] * 100)
    
    sim_d1 = round(today_risk * 0.80 + final_sim_risk * 0.20)
    sim_d3 = round(today_risk * 0.45 + final_sim_risk * 0.55)
    sim_d5 = round(today_risk * 0.18 + final_sim_risk * 0.82)
    sim_d7 = final_sim_risk
    
    sim_trajectory = [
        {"day": "Today", "risk_pct": max(4, sim_d1)},
        {"day": "Day 3", "risk_pct": max(4, sim_d3)},
        {"day": "Day 5", "risk_pct": max(4, sim_d5)},
        {"day": "Day 7", "risk_pct": max(4, sim_d7)}
    ]
    
    # Baseline trajectory
    base_traj = orig.get("risk_trajectory", [])
    if not base_traj:
        base_traj = [
            {"day": "Today", "risk_pct": today_risk},
            {"day": "Day 3", "risk_pct": min(99, round(today_risk * 1.05))},
            {"day": "Day 5", "risk_pct": min(99, round(today_risk * 1.10))},
            {"day": "Day 7", "risk_pct": min(99, round(today_risk * 1.15))}
        ]

    return {
        "field_id": sim.field_id,
        "original": {
            "predicted_yield": orig["predicted_yield"],
            "yield_loss_percentage": orig["yield_loss_percentage"],
            "failure_probability": orig["failure_probability"],
            "risk_level": orig["risk_level"],
            "risk_trajectory": base_traj
        },
        "simulated": {
            "predicted_yield": sim_res["predicted_yield"],
            "yield_loss_percentage": sim_res["yield_loss_percentage"],
            "failure_probability": sim_res["failure_probability"],
            "risk_level": sim_res["risk_level"],
            "shap_breakdown": sim_res["shap_breakdown"],
            "risk_trajectory": sim_trajectory
        },
        "deltas": {
            "yield_gain_tons": yield_delta,
            "risk_prob_reduction": abs(min(0.0, risk_prob_delta)),
            "loss_reduction_pct": max(0.0, loss_reduction_pct),
            "revenue_saved_per_acre": revenue_saved_per_acre,
            "treatment_cost_per_acre": treatment_cost_per_acre,
            "net_benefit_per_acre": net_benefit_per_acre,
            "roi_multiple": roi_multiple
        }
    }

@app.get("/api/export/{prediction_id}")
async def export_prediction_csv(prediction_id: str):
    """
    Downloads prediction CSV formatted strictly as defined in Section 12 of spec:
    field_id, prediction_date, predicted_yield, expected_yield, yield_loss_percentage,
    failure_probability, risk_level, primary_risk_factor, secondary_risk_factor,
    intervention_window, model_version
    """
    if prediction_id not in PREDICTIONS_STORE:
        raise HTTPException(status_code=404, detail="Prediction run not found.")
        
    records = PREDICTIONS_STORE[prediction_id]["results"]
    df_out = pd.DataFrame(records)
    
    # Exact required columns from specification Section 12
    export_cols = [
        "field_id", "prediction_date", "predicted_yield", "expected_yield",
        "yield_loss_percentage", "failure_probability", "risk_level",
        "primary_risk_factor", "secondary_risk_factor", "intervention_window", "model_version"
    ]
    
    for c in export_cols:
        if c not in df_out.columns:
            df_out[c] = "N/A"
            
    df_export = df_out[export_cols]
    
    stream = io.StringIO()
    df_export.to_csv(stream, index=False)
    stream.seek(0)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=CropNow_Early_Warning_Predictions_{prediction_id}.csv"
    return response

@app.get("/api/scenario/{scenario_name}")
async def load_demo_scenario(scenario_name: str):
    """Loads one or all of the ready-to-test demo datasets generated in the workspace."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_map = {
        "standard": os.path.join(base_dir, "sample_field_observations.csv"),
        "pdf": os.path.join(base_dir, "sample_field_observations.pdf"),
        "custom_columns": os.path.join(base_dir, "sample_different_column_names.csv"),
        "outbreak": os.path.join(base_dir, "sample_high_risk_outbreak.csv"),
        "training": os.path.join(base_dir, "sample_historical_harvest_training.csv")
    }
    
    if scenario_name == "all_merged":
        dfs = []
        for name, path in file_map.items():
            if name == "training":
                continue  # training has actual_yield, avoid target leakage
            if os.path.exists(path):
                if path.endswith(".pdf"):
                    with open(path, "rb") as f:
                        raw = extract_dataframe_from_pdf(f.read())
                else:
                    raw = pd.read_csv(path)
                m = auto_detect_column_mapping(list(raw.columns))
                ren = {v: k for k, v in m.items() if v and v in raw.columns}
                dfs.append(clean_and_impute_data(raw.rename(columns=ren)))
        
        merged_df = pd.concat(dfs, ignore_index=True)
        if "field_id" in merged_df.columns and "date" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["field_id", "date"], keep="last")
        else:
            merged_df = merged_df.drop_duplicates(keep="last")
            
        dataset_id = str(uuid.uuid4())
        DATASETS_STORE[dataset_id] = merged_df
        val_rep = validate_dataframe(merged_df)
        VALIDATION_STORE[dataset_id] = val_rep
        
        return {
            "dataset_id": dataset_id,
            "filename": "All 4 Scenarios Merged (CSV + PDF)",
            "format": "Multi-File Batch (PDF + 3 CSVs)",
            "total_rows": len(merged_df),
            "columns": list(merged_df.columns),
            "column_mapping_suggestion": auto_detect_column_mapping(list(merged_df.columns)),
            "validation_report": val_rep
        }
        
    file_path = file_map.get(scenario_name)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Scenario file not found.")
        
    filename = os.path.basename(file_path)
    if filename.endswith(".pdf"):
        with open(file_path, "rb") as f:
            df = extract_dataframe_from_pdf(f.read())
    else:
        df = pd.read_csv(file_path)
        
    dataset_id = str(uuid.uuid4())
    cleaned_df = clean_and_impute_data(df)
    DATASETS_STORE[dataset_id] = cleaned_df
    
    mapping_suggestion = auto_detect_column_mapping(list(cleaned_df.columns))
    validation_report = validate_dataframe(cleaned_df)
    VALIDATION_STORE[dataset_id] = validation_report
    
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "format": "PDF" if filename.endswith(".pdf") else "CSV",
        "total_rows": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "column_mapping_suggestion": mapping_suggestion,
        "validation_report": validation_report
    }

@app.post("/api/train")
async def trigger_retraining(file: Optional[UploadFile] = File(None)):
    """Retrains models with historical ground-truth harvest yield data and returns full metrics."""
    if file:
        content = await file.read()
        train_df = pd.read_csv(io.BytesIO(content))
    else:
        train_csv_path = os.path.join(DATA_DIR, "historical_training_data.csv")
        train_df = pd.read_csv(train_csv_path)
        
    validation = validate_dataframe(train_df, is_training=True)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Training dataset invalid: {validation['summary']}")
        
    metrics = engine.train_and_evaluate(train_df)
    
    # Refresh demo predictions with new model weights
    sample_df = DATASETS_STORE.get("demo-84-fields")
    if sample_df is not None:
        results_df, portfolio_summary = engine.predict_field_observations(sample_df)
        PREDICTIONS_STORE["demo-84-fields"] = {
            "results": results_df.to_dict(orient="records"),
            "portfolio_summary": portfolio_summary,
            "dataset_id": "demo-84-fields",
            "model_version": engine.version
        }
        
    return {
        "status": "success",
        "message": "Model successfully retrained and calibrated.",
        "metrics": metrics
    }

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Returns model versions, benchmark comparison, metrics, and feature importance."""
    return engine.metrics

@app.get("/api/sample-csv")
async def download_sample_csv(type: str = Query("prediction", enum=["prediction", "training"])):
    """Downloads sample template CSVs."""
    if type == "training":
        file_path = os.path.join(DATA_DIR, "historical_training_data.csv")
        filename = "CropNow_Historical_Training_Sample.csv"
    else:
        file_path = os.path.join(DATA_DIR, "sample_agricultural_data.csv")
        filename = "CropNow_Field_Observations_Template.csv"
        
    if not os.path.exists(file_path):
        train_df, sample_df = generate_agricultural_dataset()
        train_df.to_csv(os.path.join(DATA_DIR, "historical_training_data.csv"), index=False)
        sample_df.to_csv(os.path.join(DATA_DIR, "sample_agricultural_data.csv"), index=False)
        
    return FileResponse(path=file_path, filename=filename, media_type="text/csv")

# Mount frontend static files
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

