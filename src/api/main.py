
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest,
    PredictResponse,
    TripListResponse
)

from .predictor import (
    list_available_trips,
    predict_trip
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/api.log")
    ]
)

logger = logging.getLogger("PathSnap")


# Create FastAPI application
app = FastAPI(
    title="PathSnap GPS Map Matching API",
    description="GPS trajectory map matching using machine learning",
    version="1.0.0"
)


# Allow UI to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your UI origin before deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - start_time) * 1000

    logger.info(
        "%s %s | Status: %s | Latency: %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        latency
    )

    response.headers["X-Process-Time-ms"] = str(round(latency, 2))

    return response


# Root endpoint
@app.get("/")
def home():
    return {
        "project": "PathSnap",
        "message": "GPS Map Matching API is running",
        "status": "success"
    }


# List available development trips
@app.get("/trips", response_model=TripListResponse)
def get_trips():
    try:
        trips = list_available_trips()

        return {
            "trips": trips
        }

    except Exception:
        logger.exception("Failed to retrieve trip list")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve trips"
        )


# Predict road segments for a trip
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    start_time = time.perf_counter()

    try:
        result = predict_trip(request.trip_id)

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Prediction completed for trip %s",
            result["trip_id"]
        )

        return {
            "trip_id": result["trip_id"],
            "pred_coords": result["pred_coords"],
            "gps_coords": result["gps_coords"],
            "top1_acc": result["top1_acc"],
            "continuity": result["continuity"],
            "latency_ms": round(latency, 2)
        }

    except FileNotFoundError as error:
        logger.warning("Trip not found: %s", request.trip_id)

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:
        logger.warning("Invalid trip data: %s", error)

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception(
            "Prediction failed for trip %s",
            request.trip_id
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )

