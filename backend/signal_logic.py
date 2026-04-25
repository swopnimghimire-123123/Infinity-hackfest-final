# ---------------------------------------------------------------------------
# Signal sequence & timing constants
# ---------------------------------------------------------------------------

SIGNAL_SEQUENCE = [
    {"phase": "NS_straight", "axis": "NS", "movement": "straight", "duration": 7},
    {"phase": "NS_left",     "axis": "NS", "movement": "left",     "duration": 5},
    {"phase": "EW_straight", "axis": "EW", "movement": "straight", "duration": 7},
    {"phase": "EW_left",     "axis": "EW", "movement": "left",     "duration": 5},
]

YELLOW_DURATION    = 3.0   # seconds of yellow before all-red
ALL_RED_DURATION   = 2.0   # seconds all signals are red (intersection clearance)
RL_MIN_GREEN_STEPS = 10    # minimum steps before RL can switch
RL_MAX_GREEN_STEPS = 60    # force switch after this many steps

# Normalization bounds for RL state vector (must match training environment)
# [NS_straight_q, EW_straight_q, NS_left_q, EW_left_q, phase_idx_norm, elapsed_norm]
RL_STATE_QUEUE_MAX   = 60.0   # clamp queue values to this before normalizing
RL_STATE_ELAPSED_MAX = float(RL_MAX_GREEN_STEPS)

# ---------------------------------------------------------------------------
# Phase index helpers
# ---------------------------------------------------------------------------

_PHASE_INDEX: dict[str, int] = {s["phase"]: i for i, s in enumerate(SIGNAL_SEQUENCE)}


def _phase_index(phase: str) -> int:
    return _PHASE_INDEX.get(phase, 0)


def _next_phase(phase: str) -> str:
    idx = (_phase_index(phase) + 1) % len(SIGNAL_SEQUENCE)
    return SIGNAL_SEQUENCE[idx]["phase"]


# ---------------------------------------------------------------------------
# RL state normalization
# ---------------------------------------------------------------------------

def normalize_rl_state(state: list[float]) -> list[float]:
    """
    Normalize a raw 6-element RL state vector to [0, 1] range.

    Raw state layout:
        [NS_straight_queue, EW_straight_queue, NS_left_queue, EW_left_queue,
         phase_index (0-3), elapsed_steps]

    Without normalization, large queue values (e.g. 20+) cause activation
    saturation in the first hidden layer, producing near-identical Q-values
    and degrading the switching policy to near-random behaviour.
    """
    if not state or len(state) < 6:
        return state

    normed = list(state)
    # Queue counts — clamp then normalize
    for i in range(4):
        normed[i] = min(float(state[i]), RL_STATE_QUEUE_MAX) / RL_STATE_QUEUE_MAX
    # Phase index — already 0-3, normalize to [0, 1]
    normed[4] = float(state[4]) / (len(SIGNAL_SEQUENCE) - 1)
    # Elapsed steps
    normed[5] = min(float(state[5]), RL_STATE_ELAPSED_MAX) / RL_STATE_ELAPSED_MAX

    return normed


# ---------------------------------------------------------------------------
# RL pressure helpers
# ---------------------------------------------------------------------------

def rl_phase_pressure(state: list[float] | None, current_phase: str) -> float:
    """Return the raw queue pressure for the currently active phase."""
    if not state or len(state) < 4:
        return 0.0
    queue_by_phase = {
        "NS_straight": float(state[0]),
        "EW_straight": float(state[1]),
        "NS_left":     float(state[2]),
        "EW_left":     float(state[3]),
    }
    return queue_by_phase.get(current_phase, 0.0)


def rl_best_alternative_pressure(state: list[float] | None, current_phase: str) -> float:
    """Return the highest queue pressure among all phases except the active one."""
    if not state or len(state) < 4:
        return 0.0
    pressures = [float(state[i]) for i in range(4)]
    phase_order = ["NS_straight", "EW_straight", "NS_left", "EW_left"]
    active_idx = phase_order.index(current_phase) if current_phase in phase_order else -1
    if active_idx >= 0:
        pressures[active_idx] = -1.0   # exclude current phase
    return max(pressures)


def rl_target_phase(state: list[float] | None, current_phase: str) -> str:
    """Pick the highest-pressure phase from RL state, falling back to current phase."""
    if not state or len(state) < 4:
        return current_phase
    queue_by_phase = {
        "NS_straight": float(state[0]),
        "EW_straight": float(state[1]),
        "NS_left":     float(state[2]),
        "EW_left":     float(state[3]),
    }
    best_phase = current_phase
    best_value = queue_by_phase.get(current_phase, 0.0)
    for phase, value in queue_by_phase.items():
        if value > best_value:
            best_phase = phase
            best_value = value
    return best_phase


def rl_should_switch(
    state: list[float] | None,
    current_phase: str,
    requested_action: int,
) -> bool:
    """
    Gate the RL switch request: only allow a switch when the active queue
    is not meaningfully heavier than the best alternative.

    A switch is suppressed when:
        current_pressure > 1  AND  current_pressure > best_alternative + 1

    This prevents the model from abandoning a busy phase just because
    it has learned a slight preference for switching at certain step counts.
    """
    if int(requested_action) != 1:
        return False
    current_pressure  = rl_phase_pressure(state, current_phase)
    best_alternative  = rl_best_alternative_pressure(state, current_phase)
    # Allow switch unless current phase is clearly the most congested
    return current_pressure <= 1.0 or current_pressure <= best_alternative + 1.0


# ---------------------------------------------------------------------------
# RL signal tick
# ---------------------------------------------------------------------------

def next_signal_rl(
    current_phase: str,
    phase_type: str,
    elapsed: float,
    dt: float,
    requested_action: int,
    state: list[float] | None = None,
) -> dict:
    """
    Apply one RL control step while preserving yellow / all-red clearance.

    Args:
        current_phase:    Active phase name (e.g. 'NS_straight').
        phase_type:       'green' | 'yellow' | 'all_red'.
        elapsed:          Seconds (or steps) spent in the current phase_type.
        dt:               Time step size.
        requested_action: 0 = hold, 1 = request switch.
        state:            Raw 6-element RL state vector (unnormalized).

    Returns:
        {"phase": str, "phase_type": str, "elapsed": float}
    """
    elapsed += dt

    # ── All-red clearance ────────────────────────────────────────────────
    if phase_type == "all_red":
        if elapsed < ALL_RED_DURATION:
            return {"phase": current_phase, "phase_type": "all_red", "elapsed": elapsed}
        # With RL state available, return to the selected target phase directly
        # instead of forcing a fixed cyclic next phase.
        if state is not None and len(state) >= 4:
            return {
                "phase":      current_phase,
                "phase_type": "green",
                "elapsed":    0.0,
            }
        return {
            "phase":      _next_phase(current_phase),
            "phase_type": "green",
            "elapsed":    0.0,
        }

    # ── Yellow transition ────────────────────────────────────────────────
    if phase_type == "yellow":
        if elapsed < YELLOW_DURATION:
            return {"phase": current_phase, "phase_type": "yellow", "elapsed": elapsed}
        return {"phase": current_phase, "phase_type": "all_red", "elapsed": 0.0}

    # ── Green: evaluate action ───────────────────────────────────────────
    action = int(requested_action)

    # Hard bounds override the model
    if elapsed < RL_MIN_GREEN_STEPS:
        action = 0                      # too early — always hold
    elif elapsed >= RL_MAX_GREEN_STEPS:
        action = 1                      # too long  — always switch

    # Pressure gate: don't switch away from the most congested phase
    if action == 1 and not rl_should_switch(state, current_phase, action):
        action = 0

    if action == 1:
        target_phase = rl_target_phase(state, current_phase)
        return {"phase": target_phase, "phase_type": "yellow", "elapsed": 0.0}

    return {"phase": current_phase, "phase_type": "green", "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Phase parsing helpers
# ---------------------------------------------------------------------------

def parse_signal_phase(phase: str) -> dict:
    """Split 'NS_straight' → {'axis': 'NS', 'movement': 'straight'}."""
    parts    = phase.split("_", 1)
    axis     = parts[0] if len(parts) > 0 else "NS"
    movement = parts[1] if len(parts) > 1 else "straight"
    return {"axis": axis, "movement": movement}


def get_signal_stage(phase: str) -> dict:
    """Return the SIGNAL_SEQUENCE entry for the given phase name."""
    for stage in SIGNAL_SEQUENCE:
        if stage["phase"] == phase:
            return stage
    return SIGNAL_SEQUENCE[0]


# ---------------------------------------------------------------------------
# Light-head state builder
# ---------------------------------------------------------------------------

def get_light_states(phase: str, phase_type: str) -> dict:
    """
    Build per-direction signal-head dicts for the current phase & phase_type.

    Returns four heads (R1–R4) where:
        R1, R2  — North / South heads
        R3, R4  — East  / West  heads

    phase_type:
        'green'   — active axis shows green / arrow; opposing shows red.
        'yellow'  — active axis shows yellow; opposing shows red.
        'all_red' — every head shows red (intersection clearance).
    """
    stage     = parse_signal_phase(phase)
    ns_active = stage["axis"] == "NS"
    ew_active = stage["axis"] == "EW"

    def head(is_active: bool) -> dict:
        if phase_type == "all_red":
            return {
                "red": True, "yellow": False, "green": False,
                "arrow_left": False, "arrow_rs": False,
            }
        if phase_type == "yellow":
            return {
                "red": not is_active, "yellow": is_active, "green": False,
                "arrow_left": False, "arrow_rs": False,
            }
        # green
        return {
            "red":        not is_active,
            "yellow":     False,
            "green":      is_active,
            "arrow_left": is_active and stage["movement"] == "left",
            "arrow_rs":   is_active and stage["movement"] == "straight",
        }

    ns_head = head(ns_active)
    ew_head = head(ew_active)

    return {
        "R1": ns_head,
        "R2": ns_head,   # N/S always identical — kept for API compatibility
        "R3": ew_head,
        "R4": ew_head,   # E/W always identical — kept for API compatibility
    }


# ---------------------------------------------------------------------------
# Vehicle permission check
# ---------------------------------------------------------------------------

def can_vehicle_proceed(
    signal_phase: str,
    phase_type: str,
    vehicle_axis: str,
    vehicle_maneuver: str,
) -> bool:
    """
    Return True only when the intersection is green and the vehicle's
    axis + maneuver match the active movement.
    """
    if phase_type != "green":
        return False

    stage = parse_signal_phase(signal_phase)
    if stage["axis"] != vehicle_axis:
        return False
    if stage["movement"] == "straight":
        return vehicle_maneuver in ("straight", "right")
    if stage["movement"] == "left":
        return vehicle_maneuver == "left"
    return False


# ---------------------------------------------------------------------------
# Signal state-machine tick  (fixed / adaptive / wave strategies)
# ---------------------------------------------------------------------------

def next_signal(
    current_phase: str,
    phase_type: str,
    elapsed: float,
    strategy: str,
    queues_ns: float,
    queues_ew: float,
    approaching_ns: float,
    approaching_ew: float,
    dt: float,
) -> dict:
    """
    Advance the signal state machine by dt seconds.

    Transition path:
        green → yellow (YELLOW_DURATION s) → all_red (ALL_RED_DURATION s) → next green

    Returns:
        {"phase": str, "phase_type": str, "elapsed": float}
    """
    elapsed += dt

    # ── All-red clearance ────────────────────────────────────────────────
    if phase_type == "all_red":
        if elapsed < ALL_RED_DURATION:
            return {"phase": current_phase, "phase_type": "all_red", "elapsed": elapsed}
        return {
            "phase":      _next_phase(current_phase),
            "phase_type": "green",
            "elapsed":    0.0,
        }

    # ── Yellow transition ────────────────────────────────────────────────
    if phase_type == "yellow":
        if elapsed < YELLOW_DURATION:
            return {"phase": current_phase, "phase_type": "yellow", "elapsed": elapsed}
        return {"phase": current_phase, "phase_type": "all_red", "elapsed": 0.0}

    # ── Green: compute hold duration ─────────────────────────────────────
    stage  = get_signal_stage(current_phase)
    parsed = parse_signal_phase(current_phase)

    queue_pressure = (
        queues_ns + approaching_ns * 0.5
        if parsed["axis"] == "NS"
        else queues_ew + approaching_ew * 0.5
    )

    if strategy == "wave":
        bonus = 4.0
    elif strategy == "adaptive":
        bonus = min(3.0, queue_pressure * 0.6)
    else:
        bonus = 0.0

    hold_duration = stage["duration"] + bonus

    if elapsed < hold_duration:
        return {"phase": current_phase, "phase_type": "green", "elapsed": elapsed}

    # Green expired → enter yellow
    return {"phase": current_phase, "phase_type": "yellow", "elapsed": 0.0}


# ---------------------------------------------------------------------------
# DQN forward pass  (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def rl_action_from_weights(weights: dict, state: list[float]) -> int:
    """
    Run a pure-Python forward pass through the exported DQN weight dict.

    IMPORTANT: the state vector is normalized before inference so that
    queue magnitudes don't saturate the hidden-layer activations.  The
    model was trained on normalized inputs; skipping this step causes
    near-identical Q-values and effectively random switching behaviour.

    Args:
        weights: {"layers": [{"W": [...], "b": [...]}, ...]}
        state:   Raw 6-element RL state vector.

    Returns:
        0 (hold) or 1 (switch).
    """
    # Normalize before inference
    x = normalize_rl_state([float(v) for v in state])

    last = len(weights["layers"]) - 1
    for layer_index, layer in enumerate(weights["layers"]):
        w    = layer["W"]
        b    = layer["b"]
        rows = len(x)
        cols = len(b)
        out  = []
        for j in range(cols):
            acc = b[j]
            for i in range(rows):
                acc += x[i] * w[i * cols + j]
            # ReLU on hidden layers; linear on output layer
            out.append(acc if layer_index == last else max(0.0, acc))
        x = out

    return 1 if x[1] > x[0] else 0