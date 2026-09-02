import uvicorn
import os
import sys

# Ensure current workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=================================================================")
    print("[CropNow] AI Crop Failure & Yield-Loss Early Warning System")
    print("[CropNow] Starting FastAPI Server on http://127.0.0.1:8000 ...")
    print("=================================================================")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
