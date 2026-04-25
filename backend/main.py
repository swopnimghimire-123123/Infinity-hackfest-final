from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from signal_logic import (
    next_signal,
    next_signal_rl,
    rl_action_from_weights,
    get_light_states,
    can_vehicle_proceed,
    normalize_rl_state,
    SIGNAL_SEQUENCE,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Traffic Signal Controller", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RL_MODEL: Optional[dict] = None
RL_CONGESTION_CAP = 32.0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Queues(BaseModel):
    NS: float = 0.0
    EW: float = 0.0


class Approaching(BaseModel):
    NS: float = 0.0
    EW: float = 0.0


class SignalTickRequest(BaseModel):
    current_phase: str   = "NS_straight"
    phase_type:    str   = "green"        # "green" | "yellow" | "all_red"
    elapsed:       float = 0.0
    dt:            float = 0.25
    strategy:      str   = "adaptive"     # "fixed" | "adaptive" | "wave" | "rl"
    queues:        Queues      = Field(default_factory=Queues)
    approaching:   Approaching = Field(default_factory=Approaching)
    # Legacy ML fields (kept for backward compatibility, unused)
    ml_snapshot:   Optional[dict]       = None
    ml_state:      Optional[dict]       = None
    # RL state: [NS_straight_q, EW_straight_q, NS_left_q, EW_left_q, phase_idx, elapsed]
    rl_state:      Optional[List[float]] = None


class SignalTickResponse(BaseModel):
    phase:      str
    phase_type: str
    elapsed:    float
    lights:     dict
    # Optional: expose the RL action that was taken (for debugging)
    rl_action:  Optional[int] = None


class RLModelStatus(BaseModel):
    loaded:      bool
    version:     Optional[int]       = None
    stateSize:   Optional[int]       = None
    actionSize:  Optional[int]       = None
    hiddenSizes: Optional[list[int]] = None
    totalSteps:  Optional[int]       = None


class VehicleCheckRequest(BaseModel):
    signal_phase:    str
    phase_type:      str
    vehicle_axis:    str
    vehicle_maneuver: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_PHASES     = {s["phase"] for s in SIGNAL_SEQUENCE}
_VALID_PHASE_TYPES = {"green", "yellow", "all_red"}
_VALID_STRATEGIES  = {"fixed", "adaptive", "wave", "rl"}


def _validate_tick(body: SignalTickRequest) -> None:
    if body.current_phase not in _VALID_PHASES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown phase '{body.current_phase}'. "
                   f"Valid phases: {sorted(_VALID_PHASES)}",
        )
    if body.phase_type not in _VALID_PHASE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown phase_type '{body.phase_type}'. "
                   f"Valid: {sorted(_VALID_PHASE_TYPES)}",
        )
    if body.strategy not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy '{body.strategy}'. "
                   f"Valid: {sorted(_VALID_STRATEGIES)}",
        )
    if body.dt <= 0:
        raise HTTPException(status_code=422, detail="dt must be positive.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/signal/tick", response_model=SignalTickResponse)
def signal_tick(body: SignalTickRequest):
    """
    Advance the traffic signal by one time step.

    When strategy='rl' and a model is loaded, the DQN model decides
    whether to hold or switch the current phase.  The rl_state vector
    must have exactly 6 elements; otherwise the endpoint falls back to
    the adaptive strategy.
    """
    _validate_tick(body)

    global RL_MODEL
    rl_action: Optional[int] = None

    use_rl = (
        body.strategy == "rl"
        and RL_MODEL is not None
        and body.rl_state is not None
        and len(body.rl_state) == 6
    )

    if use_rl:
        rl_action = rl_action_from_weights(RL_MODEL, body.rl_state)
        total_queue = float(body.queues.NS + body.queues.EW)
        if total_queue > RL_CONGESTION_CAP:
            state = body.rl_state
            demand = {
                "NS_straight": float(state[0]),
                "EW_straight": float(state[1]),
                "NS_left": float(state[2]),
                "EW_left": float(state[3]),
            }
            target_phase = max(demand, key=demand.get)
            result = {
                "phase": target_phase,
                "phase_type": "green",
                "elapsed": 0.0,
            }
        else:
            result = next_signal_rl(
                current_phase    = body.current_phase,
                phase_type       = body.phase_type,
                elapsed          = body.elapsed,
                dt               = body.dt,
                requested_action = rl_action,
                state            = body.rl_state,
            )
    else:
        result = next_signal(
            current_phase  = body.current_phase,
            phase_type     = body.phase_type,
            elapsed        = body.elapsed,
            strategy       = body.strategy if body.strategy != "rl" else "adaptive",
            queues_ns      = body.queues.NS,
            queues_ew      = body.queues.EW,
            approaching_ns = body.approaching.NS,
            approaching_ew = body.approaching.EW,
            dt             = body.dt,
        )

    lights = get_light_states(result["phase"], result["phase_type"])

    return SignalTickResponse(
        phase      = result["phase"],
        phase_type = result["phase_type"],
        elapsed    = result["elapsed"],
        lights     = lights,
        rl_action  = rl_action,
    )


@app.get("/health")
def health():
    """Liveness check — also reports RL model status."""
    return {
        "status": "ok",
        "rl_model_loaded": RL_MODEL is not None,
        "rl_model_version": RL_MODEL.get("version") if RL_MODEL else None,
    }


@app.post("/rl/model", response_model=RLModelStatus)
def load_rl_model(model: dict):
    """
    Upload DQN weights to the server for backend inference.
    The model dict must contain a 'layers' key with W/b arrays.
    """
    global RL_MODEL

    if "layers" not in model or not isinstance(model["layers"], list):
        raise HTTPException(
            status_code=422,
            detail="Model must contain a 'layers' list with {W, b} dicts.",
        )

    RL_MODEL = model
    return RLModelStatus(
        loaded      = True,
        version     = model.get("version"),
        stateSize   = model.get("stateSize"),
        actionSize  = model.get("actionSize"),
        hiddenSizes = model.get("hiddenSizes"),
        totalSteps  = model.get("totalSteps"),
    )


@app.delete("/rl/model")
def clear_rl_model():
    """Remove the currently loaded RL model from server memory."""
    global RL_MODEL
    RL_MODEL = None
    return {"status": "cleared"}


@app.get("/rl/model", response_model=RLModelStatus)
def get_rl_model_status():
    """Report whether a model is loaded and its metadata."""
    if RL_MODEL is None:
        return RLModelStatus(loaded=False)
    return RLModelStatus(
        loaded      = True,
        version     = RL_MODEL.get("version"),
        stateSize   = RL_MODEL.get("stateSize"),
        actionSize  = RL_MODEL.get("actionSize"),
        hiddenSizes = RL_MODEL.get("hiddenSizes"),
        totalSteps  = RL_MODEL.get("totalSteps"),
    )


@app.post("/rl/normalize")
def normalize_state(state: List[float]):
    """
    Debug endpoint: returns the normalized form of a raw RL state vector.
    Useful for verifying that frontend and backend normalization agree.
    """
    if len(state) != 6:
        raise HTTPException(status_code=422, detail="State must have exactly 6 elements.")
    return {"raw": state, "normalized": normalize_rl_state(state)}


@app.post("/vehicle/check")
def vehicle_check(body: VehicleCheckRequest):
    """Check whether a vehicle may proceed given the current signal state."""
    allowed = can_vehicle_proceed(
        signal_phase     = body.signal_phase,
        phase_type       = body.phase_type,
        vehicle_axis     = body.vehicle_axis,
        vehicle_maneuver = body.vehicle_maneuver,
    )
    return {"can_proceed": allowed}


@app.get("/phases")
def list_phases():
    """Return the full signal phase sequence (useful for frontend initialisation)."""
    return {"phases": SIGNAL_SEQUENCE}