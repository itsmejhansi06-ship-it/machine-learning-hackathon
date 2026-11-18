
from typing import Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    trip_id: str


class PredictResponse(BaseModel):
    trip_id: str
    pred_coords: list[list[float]]
    gps_coords: list[list[float]]
    top1_acc: Optional[float] = None
    continuity: Optional[float] = None
    latency_ms: float


class TripInfo(BaseModel):
    trip_id: str
    split: str


class TripListResponse(BaseModel):
    trips: list[TripInfo]

