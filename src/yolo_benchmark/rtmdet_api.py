"""FastAPI endpoint for the selected RTMDet-x Top-1 model."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .rtmdet_x import RTMDetXPredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class PredictionResponse(BaseModel):
    model: Literal["rtmdet-x"]
    top_k: Literal[1]
    prediction: Literal["computer", "book", "other"]
    confidence: float
    group_scores: dict[str, float]
    decision_reason: Literal["highest_passing_group", "no_passing_detection"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = RTMDetXPredictor()
    yield
    del app.state.predictor


app = FastAPI(title="RTMDet-x Top-1 inference", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "rtmdet-x", "top_k": 1}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG, PNG or another OpenCV-decodable image")],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> PredictionResponse:
    expected_key = os.getenv("RTMDET_API_KEY")
    if expected_key and not compare_digest(x_api_key or "", expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        raise HTTPException(status_code=400, detail="Invalid image") from None
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    result = await run_in_threadpool(request.app.state.predictor.predict, image)
    return PredictionResponse(**result)
