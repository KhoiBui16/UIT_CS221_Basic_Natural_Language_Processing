# app/app_fastapi.py
import sys
import os

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from typing import List, Optional, Any
import torch
import pandas as pd
from inference_module import predict_hallucination

app = FastAPI(
    title="Hallucination Detection API",
    description="Backend API cho mô hình phát hiện ảo giác LLM tiếng Việt",
    version="2.2",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "KhoiBui/xlm-roberta-large-hallucination-classification"


class SingleRequest(BaseModel):
    id: Optional[Any] = None
    context: str
    prompt: str
    response: str

    model_config = ConfigDict(extra="allow")


class BatchRequest(BaseModel):
    data: List[SingleRequest]
    model_name: str = DEFAULT_MODEL


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="Empty data list")

    try:
        data_list = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in request.data
        ]
        df = pd.DataFrame(data_list)

        print(f"📨 Batch received: {len(df)} samples. Model: {request.model_name}")
        result_df = predict_hallucination(df, request.model_name, device_str=DEVICE)

        return result_df.to_dict(orient="records")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Đứng tại thư mục gốc
# uvicorn app.app_fastapi:app --host 0.0.0.0 --port 8095 --reload
