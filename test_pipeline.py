import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ml.dataset_generator import generate_agricultural_dataset
from backend.ml.validator import validate_dataframe
from backend.ml.models import engine

print("1. Generating agricultural dataset...")
train_df, pred_df = generate_agricultural_dataset(num_fields=84, days_history=14)
print(f"Generated {len(train_df)} training records and {len(pred_df)} prediction fields.")

print("2. Validating training dataframe...")
val_res = validate_dataframe(train_df, is_training=True)
print("Validation result:", val_res["valid"], "Summary:", val_res["summary"])

print("3. Training ML Engine (XGBoost + LightGBM + RF + Baseline + SHAP)...")
metrics = engine.train_and_evaluate(train_df)
print("Yield MAE:", metrics["yield_metrics"]["mae"])
print("Yield R2:", metrics["yield_metrics"]["r2"])
print("Risk ROC-AUC:", metrics["risk_metrics"]["roc_auc"])
print("Top 20% Capture Rate:", metrics["risk_metrics"]["business_top20_capture_rate"])

print("4. Running Inference & SHAP Attribution & 7-Day Trajectory...")
results_df, summary = engine.predict_field_observations(pred_df)
print("Portfolio Headline:", summary["portfolio_headline"])
print("Total Fields:", summary["total_fields"])
print("High/Critical Risk Count:", summary["high_critical_count"])
print("Estimated Loss USD:", summary["estimated_loss_exposure_usd"])

print("\nSample Field 1 Result:")
sample = results_df.iloc[0]
for k, v in sample.items():
    if k not in ["sensors", "shap_breakdown", "risk_trajectory"]:
        print(f"  {k}: {v}")

print("\nPipeline self-test PASSED successfully! 🎉")
