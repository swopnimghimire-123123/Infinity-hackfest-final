import { useEffect, useMemo, useRef, useState } from 'react'
import { FaArrowDown, FaArrowLeft, FaArrowRight, FaArrowUp } from 'react-icons/fa6'
import './App.css'
import { useRLTrainer }    from './rl/useRLTrainer.js'
import { useRLController } from './rl/useRLController.js'
import { runValidation }   from './rl/validate.js'
import { formatComparisonTable, improvementPercent } from './rl/metrics.js'
import { clearModel, exportToFile, importFromFile } from './rl/modelStorage.js'
import { PHASES as RL_PHASES, MIN_GREEN_STEPS, MAX_GREEN_STEPS, STATE_QUEUE_NORM } from './rl/trafficEnv.js'

const WORLD = {
  width: 980,
  height: 580,
  road: 180,
  laneOffset: 32,
}

const TRAINING_PRESETS = {
  quick: {
    label: 'Quick Demo',
    episodes: 80,
    epsilonDecayEpisodes: 70,
    progressInterval: 5,
    convergenceWindow: 12,
    hiddenSizes: [48, 48],
    lr: 0.0006,
    gamma: 0.985,
    trainingStartSteps: 1500,
    bufferSize: 20000,
    batchSize: 96,
    targetUpdateFreq: 500,
    trainEvery: 2,
    spawnRateMin: 0.5,
    spawnRateMax: 0.92,
    leftTurnMultMin: 0.75,
    leftTurnMultMax: 1.6,
    enableEarlyStopping: true,
  },
  balanced: {
    label: 'Balanced',
    episodes: 200,
    epsilonDecayEpisodes: 180,
    progressInterval: 10,
    convergenceWindow: 20,
    hiddenSizes: [64, 64],
    lr: 0.00045,
    gamma: 0.99,
    trainingStartSteps: 2500,
    bufferSize: 40000,
    batchSize: 128,
    targetUpdateFreq: 700,
    trainEvery: 2,
    spawnRateMin: 0.55,
    spawnRateMax: 0.96,
    leftTurnMultMin: 0.7,
    leftTurnMultMax: 1.8,
    enableEarlyStopping: true,
  },
  thorough: {
    label: 'Thorough',
    episodes: 350,
    epsilonDecayEpisodes: 300,
    progressInterval: 20,
    convergenceWindow: 30,
    hiddenSizes: [64, 64],
    lr: 0.00035,
    gamma: 0.99,
    trainingStartSteps: 3000,
    bufferSize: 50000,
    batchSize: 128,
    targetUpdateFreq: 800,
    trainEvery: 2,
    spawnRateMin: 0.6,
    spawnRateMax: 0.98,
    leftTurnMultMin: 0.7,
    leftTurnMultMax: 1.8,
    // Thorough is intended for stable final models: always run full episode budget.
    enableEarlyStopping: false,
  },
}

const DIRECTIONS = ['N2S', 'S2N', 'W2E', 'E2W']

const BASE_SPEED = 110
const VEHICLE_LENGTH = 24
const JOINT_GAP = 14
const SAFE_GAP = 32
const LANE_SPLIT = 24
const VEHICLE_SOLID_RADIUS = 13
const STUCK_RECOVERY_SECONDS = 3.5
const STUCK_FAILSAFE_SECONDS = 9
const STUCK_NUDGE_DISTANCE = 10
const TAXI_ASPECT_RATIO = 660 / 980
const VEHICLE_SPRITES = {
  truck: '/truck.png',
  taxi: '/taxi.png',
  bike: '/bike.png',
}
const VEHICLE_TYPE_WEIGHTS_KATHMANDU = [
  { type: 'truck', weight: 0.34 },
  { type: 'taxi',  weight: 0.41 },
  { type: 'bike',  weight: 0.25 },
]
const VEHICLE_TYPE_WEIGHTS_CAR_HEAVY = [
  { type: 'truck', weight: 0.22 },
  { type: 'taxi',  weight: 0.70 },
  { type: 'bike',  weight: 0.08 },
]

const PCU_BY_TYPE = { truck: 3, taxi: 1, bike: 0.5 }

const VEHICLE_RENDER_DIMS = {
  truck: {
    lead:  { length: VEHICLE_LENGTH + 25, width: 28 },
    trail: { length: VEHICLE_LENGTH + 20, width: 24.8 },
  },
  taxi: {
    lead:  { length: VEHICLE_LENGTH + 14, width: (VEHICLE_LENGTH + 14) * TAXI_ASPECT_RATIO },
    trail: { length: VEHICLE_LENGTH + 11, width: (VEHICLE_LENGTH + 11) * TAXI_ASPECT_RATIO },
  },
  bike: {
    lead:  { length: VEHICLE_LENGTH + 12, width: 16 },
    trail: { length: VEHICLE_LENGTH + 9,  width: 14 },
  },
}

const VEHICLE_HEADING_OFFSETS = {
  truck: 0,
  taxi:  Math.PI / 2,
  bike:  -Math.PI / 2,
}

function getVehicleWeightsForMix(vehicleMix) {
  if (vehicleMix === 'car_heavy') return VEHICLE_TYPE_WEIGHTS_CAR_HEAVY
  return VEHICLE_TYPE_WEIGHTS_KATHMANDU
}

function pickVehicleType(vehicleMix) {
  const weights = getVehicleWeightsForMix(vehicleMix)
  const roll = Math.random()
  let acc = 0
  for (const entry of weights) {
    acc += entry.weight
    if (roll <= acc) return entry.type
  }
  return 'taxi'
}

function getVehicleRenderDims(vehicleType, isLeadSegment) {
  const preset = VEHICLE_RENDER_DIMS[vehicleType] ?? VEHICLE_RENDER_DIMS.taxi
  return isLeadSegment ? preset.lead : preset.trail
}

function getVehicleCollisionRadius(vehicleType) {
  const leadDims = getVehicleRenderDims(vehicleType, true)
  const scaled = Math.max(leadDims.length, leadDims.width) * 0.37
  return Math.max(11, Math.min(21, scaled))
}

function getFollowingGap(follower, leader) {
  const followerLen = getVehicleRenderDims(follower.type, true).length
  const leaderLen   = getVehicleRenderDims(leader.type,   true).length
  return SAFE_GAP + (followerLen + leaderLen) * 0.16
}

const SIGNAL_SEQUENCE = [
  { phase: 'NS_straight', axis: 'NS', movement: 'straight', duration: 7 },
  { phase: 'NS_left',     axis: 'NS', movement: 'left',     duration: 5 },
  { phase: 'EW_straight', axis: 'EW', movement: 'straight', duration: 7 },
  { phase: 'EW_left',     axis: 'EW', movement: 'left',     duration: 5 },
]

const YELLOW_DURATION  = 3.0
const ALL_RED_DURATION = 2.0
const RL_CONGESTION_CAP = 32
const RL_CONGESTION_RECOVERY = 26
const RL_STATE_INDEX = {
  NS_straight: 0,
  EW_straight: 1,
  NS_left: 2,
  EW_left: 3,
}

function lerp(a, b, t) { return a + (b - a) * t }
function easeInOutSine(t) { return -(Math.cos(Math.PI * t) - 1) / 2 }

function sampleLine(a, b, steps = 16) {
  const pts = []
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    pts.push({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) })
  }
  return pts
}

function sampleQuadratic(a, c, b, steps = 20) {
  const pts = []
  for (let i = 0; i <= steps; i++) {
    const t = i / steps, mt = 1 - t
    pts.push({
      x: mt * mt * a.x + 2 * mt * t * c.x + t * t * b.x,
      y: mt * mt * a.y + 2 * mt * t * c.y + t * t * b.y,
    })
  }
  return pts
}

function laneIndexOffset(laneIndex) {
  return laneIndex === 0 ? -LANE_SPLIT / 2 : LANE_SPLIT / 2
}

function lengthBetween(points, from, to) {
  let len = 0
  for (let i = from + 1; i <= to; i++) {
    const dx = points[i].x - points[i - 1].x
    const dy = points[i].y - points[i - 1].y
    len += Math.hypot(dx, dy)
  }
  return len
}

function withPathMeta(points, stopDistance) {
  const cumulative = [0]
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x
    const dy = points[i].y - points[i - 1].y
    cumulative[i] = cumulative[i - 1] + Math.hypot(dx, dy)
  }
  return { points, cumulative, length: cumulative[cumulative.length - 1], stopDistance }
}

function buildPath(direction, maneuver, laneIndex = 0) {
  const halfW = WORLD.width / 2 + 80
  const halfH = WORLD.height / 2 + 80
  const turnInset = WORLD.road / 2 + 8
  const lane = WORLD.laneOffset
  const incomingLaneDelta = laneIndexOffset(laneIndex)

  const rightTurnOutgoingLaneByDirection = { N2S: 0, S2N: 1, W2E: 0, E2W: 1 }
  const leftTurnOutgoingLaneByDirection  = { N2S: 0, S2N: 1, W2E: 0, E2W: 1 }
  const rightTurnLaneDelta = laneIndexOffset(rightTurnOutgoingLaneByDirection[direction])
  const leftTurnLaneDelta  = laneIndexOffset(leftTurnOutgoingLaneByDirection[direction])

  const incomingLaneMap = {
    N2S: { x: -lane + incomingLaneDelta, y: -halfH },
    S2N: { x:  lane + incomingLaneDelta, y:  halfH },
    W2E: { x: -halfW, y:  lane + incomingLaneDelta },
    E2W: { x:  halfW, y: -lane + incomingLaneDelta },
  }
  const straightEndMap = {
    N2S: { x: -lane + incomingLaneDelta, y:  halfH },
    S2N: { x:  lane + incomingLaneDelta, y: -halfH },
    W2E: { x:  halfW, y:  lane + incomingLaneDelta },
    E2W: { x: -halfW, y: -lane + incomingLaneDelta },
  }
  const rightTurnMap = {
    N2S: { pre: { x: -lane + incomingLaneDelta, y: -turnInset }, post: { x: -turnInset, y: -lane + rightTurnLaneDelta }, ctrl: { x: -turnInset, y: -turnInset }, end: { x: -halfW, y: -lane + rightTurnLaneDelta } },
    S2N: { pre: { x:  lane + incomingLaneDelta, y:  turnInset }, post: { x:  turnInset, y:  lane + rightTurnLaneDelta }, ctrl: { x:  turnInset, y:  turnInset }, end: { x:  halfW, y:  lane + rightTurnLaneDelta } },
    W2E: { pre: { x: -turnInset, y:  lane + incomingLaneDelta }, post: { x: -lane + rightTurnLaneDelta, y:  turnInset }, ctrl: { x: -turnInset, y:  turnInset }, end: { x: -lane + rightTurnLaneDelta, y:  halfH } },
    E2W: { pre: { x:  turnInset, y: -lane + incomingLaneDelta }, post: { x:  lane + rightTurnLaneDelta, y: -turnInset }, ctrl: { x:  turnInset, y: -turnInset }, end: { x:  lane + rightTurnLaneDelta, y: -halfH } },
  }
  const leftTurnMap = {
    N2S: { pre: { x: -lane + incomingLaneDelta, y: -turnInset }, post: { x:  turnInset, y:  lane + leftTurnLaneDelta }, ctrl: { x:  turnInset, y:  turnInset }, end: { x:  halfW, y:  lane + leftTurnLaneDelta } },
    S2N: { pre: { x:  lane + incomingLaneDelta, y:  turnInset }, post: { x: -turnInset, y: -lane + leftTurnLaneDelta }, ctrl: { x: -turnInset, y: -turnInset }, end: { x: -halfW, y: -lane + leftTurnLaneDelta } },
    W2E: { pre: { x: -turnInset, y:  lane + incomingLaneDelta }, post: { x:  lane + leftTurnLaneDelta, y: -turnInset }, ctrl: { x:  turnInset, y: -turnInset }, end: { x:  lane + leftTurnLaneDelta, y: -halfH } },
    E2W: { pre: { x:  turnInset, y: -lane + incomingLaneDelta }, post: { x: -lane + leftTurnLaneDelta, y:  turnInset }, ctrl: { x: -turnInset, y:  turnInset }, end: { x: -lane + leftTurnLaneDelta, y:  halfH } },
  }

  const start = incomingLaneMap[direction]
  if (maneuver === 'straight') {
    const end = straightEndMap[direction]
    const path = sampleLine(start, end, 64)
    const stopDistance = axisForDirection(direction) === 'NS' ? halfH - turnInset : halfW - turnInset
    return withPathMeta(path, stopDistance)
  }

  const turn = maneuver === 'left' ? leftTurnMap[direction] : rightTurnMap[direction]
  const seg1 = sampleLine(start, turn.pre, 20)
  const seg2 = sampleQuadratic(turn.pre, turn.ctrl, turn.post, 26)
  const seg3 = sampleLine(turn.post, turn.end, 20)
  const path = [...seg1, ...seg2.slice(1), ...seg3.slice(1)]
  const stopDistance = lengthBetween(path, 0, seg1.length - 1) - 2
  return withPathMeta(path, stopDistance)
}

function samplePath(path, distance) {
  const d = Math.max(0, Math.min(distance, path.length))
  let idx = 1
  while (idx < path.cumulative.length && path.cumulative[idx] < d) idx++
  const i1 = Math.max(1, idx), i0 = i1 - 1
  const span = path.cumulative[i1] - path.cumulative[i0] || 1
  const t = (d - path.cumulative[i0]) / span
  const p0 = path.points[i0], p1 = path.points[i1]
  return {
    x: lerp(p0.x, p1.x, t),
    y: lerp(p0.y, p1.y, t),
    angle: Math.atan2(p1.y - p0.y, p1.x - p0.x),
  }
}

function axisForDirection(direction) {
  return direction === 'N2S' || direction === 'S2N' ? 'NS' : 'EW'
}

function parseSignalPhase(phase) {
  const [axis = 'NS', movement = 'straight'] = phase.split('_')
  return { axis, movement }
}

function getSignalStage(phase) {
  return SIGNAL_SEQUENCE.find((s) => s.phase === phase) ?? SIGNAL_SEQUENCE[0]
}

// ---------------------------------------------------------------------------
// LOCAL FALLBACK signal state machine  (mirrors backend exactly)
// Used when the backend is unreachable.
// ---------------------------------------------------------------------------
function nextSignalLocal(signal, strategy, queues, approaching, dt) {
  const elapsed = signal.elapsed + dt
  const phaseType = signal.phase_type ?? 'green'

  if (phaseType === 'all_red') {
    if (elapsed < ALL_RED_DURATION)
      return { phase: signal.phase, phase_type: 'all_red', elapsed }
    const idx = SIGNAL_SEQUENCE.findIndex((s) => s.phase === signal.phase)
    const next = SIGNAL_SEQUENCE[(idx + 1) % SIGNAL_SEQUENCE.length]
    return { phase: next.phase, phase_type: 'green', elapsed: 0 }
  }

  if (phaseType === 'yellow') {
    if (elapsed < YELLOW_DURATION)
      return { phase: signal.phase, phase_type: 'yellow', elapsed }
    return { phase: signal.phase, phase_type: 'all_red', elapsed: 0 }
  }

  // green
  const stage = getSignalStage(signal.phase)
  const parsed = parseSignalPhase(signal.phase)
  const queuePressure =
    parsed.axis === 'NS'
      ? queues.NS + approaching.NS * 0.5
      : queues.EW + approaching.EW * 0.5
  const bonus =
    strategy === 'wave' ? 4 : strategy === 'adaptive' ? Math.min(3, queuePressure * 0.6) : 0
  if (elapsed < stage.duration + bonus)
    return { phase: signal.phase, phase_type: 'green', elapsed }
  return { phase: signal.phase, phase_type: 'yellow', elapsed: 0 }
}

// ---------------------------------------------------------------------------
// Light states — used both for canvas drawing and vehicle permission checks
// ---------------------------------------------------------------------------
function getFallbackLightsFromPhase(phase, phaseType = 'green') {
  const stage    = parseSignalPhase(phase)
  const nsActive = stage.axis === 'NS'
  const ewActive = stage.axis === 'EW'

  function head(isActive) {
    if (phaseType === 'all_red')
      return { red: true,  yellow: false, green: false, arrow_left: false, arrow_rs: false }
    if (phaseType === 'yellow')
      return { red: !isActive, yellow: isActive, green: false, arrow_left: false, arrow_rs: false }
    return {
      red:        !isActive,
      yellow:     false,
      green:      isActive,
      arrow_left: isActive && stage.movement === 'left',
      arrow_rs:   isActive && stage.movement === 'straight',
    }
  }

  return { R1: head(nsActive), R2: head(nsActive), R3: head(ewActive), R4: head(ewActive) }
}

// ---------------------------------------------------------------------------
// Vehicle permission — reads from backend lights, falls back to local logic
// ---------------------------------------------------------------------------
function canVehicleProceedFromBackend(signal, vehicle) {
  const phaseType = signal.phase_type ?? 'green'

  // During yellow or all-red nobody enters the box
  if (phaseType !== 'green') return false

  const axis = axisForDirection(vehicle.direction)
  const pole =
    axis === 'NS'
      ? vehicle.direction === 'N2S' ? 'R1' : 'R2'
      : vehicle.direction === 'W2E' ? 'R3' : 'R4'

  const lights = signal.lights?.[pole]

  if (!lights) {
    // Backend hasn't responded yet — use local phase logic
    const stage = parseSignalPhase(signal.phase ?? 'NS_straight')
    if (stage.axis !== axis) return false
    if (stage.movement === 'straight') return vehicle.maneuver === 'straight' || vehicle.maneuver === 'right'
    return vehicle.maneuver === 'left'
  }

  if (vehicle.maneuver === 'left')                                  return !!lights.arrow_left
  if (vehicle.maneuver === 'straight' || vehicle.maneuver === 'right') return !!lights.arrow_rs
  return false
}

// ---------------------------------------------------------------------------
// Queue / approach helpers
// ---------------------------------------------------------------------------
function countQueues(vehicles, signal) {
  const q = { NS: 0, EW: 0 }
  for (const v of vehicles) {
    const axis = axisForDirection(v.direction)
    if (!canVehicleProceedFromBackend(signal, v) && v.distance < v.path.stopDistance + 44)
      q[axis] += 1
  }
  return q
}

function countApproaching(vehicles, axis) {
  let n = 0
  for (const v of vehicles) {
    if (axisForDirection(v.direction) !== axis) continue
    const gapToStop = v.path.stopDistance - v.distance
    if (gapToStop > 0 && gapToStop < 140 && v.speed > BASE_SPEED * 0.7) n++
  }
  return n
}

function getPhaseFrontDemand(vehicles) {
  const demand = { NS_straight: 0, NS_left: 0, EW_straight: 0, EW_left: 0 }
  for (const v of vehicles) {
    const gapToStop = v.path.stopDistance - v.distance
    const isFrontVehicle = gapToStop > -10 && gapToStop < 90
    if (!isFrontVehicle) continue

    const axis = axisForDirection(v.direction)
    if (axis === 'NS') {
      if (v.maneuver === 'left') demand.NS_left += 1
      else demand.NS_straight += 1
    } else {
      if (v.maneuver === 'left') demand.EW_left += 1
      else demand.EW_straight += 1
    }
  }
  return demand
}

// ---------------------------------------------------------------------------
// ML snapshot helpers (unchanged)
// ---------------------------------------------------------------------------
function discretizeQueuePcu(q)    { return q < 5 ? 0 : q < 15 ? 1 : 2 }
function discretizeWaitSeconds(w) { return w < 30 ? 0 : w < 90 ? 1 : 2 }

function getTimeOfDayBin(date = new Date()) {
  const h = date.getHours()
  if (h >= 7  && h < 11) return 1
  if (h >= 17 && h < 21) return 2
  return 0
}

function getPhaseAxisCode(phase)  { return parseSignalPhase(phase).axis === 'NS' ? 0 : 1 }
function getGreenTypeCode(phase)  { return parseSignalPhase(phase).movement === 'left' ? 1 : 0 }

function getAxisMlStats(sim, axis) {
  let queuePcu = 0, queuedCount = 0, waitSum = 0
  for (const v of sim.vehicles) {
    if (axisForDirection(v.direction) !== axis) continue
    const blocked   = !canVehicleProceedFromBackend(sim.signal, v)
    const nearStop  = v.distance < v.path.stopDistance + 44
    if (blocked && nearStop) {
      queuePcu += PCU_BY_TYPE[v.type] ?? 1
      queuedCount++
      waitSum += v.wait
    }
  }
  const avgWait = queuedCount === 0 ? 0 : waitSum / queuedCount
  return {
    queuePcu,
    queuePcuBin:  discretizeQueuePcu(queuePcu),
    avgWait,
    avgWaitBin:   discretizeWaitSeconds(avgWait),
  }
}

function buildSignalMlState(sim, options = {}) {
  const ns = getAxisMlStats(sim, 'NS')
  const ew = getAxisMlStats(sim, 'EW')
  const currentPhaseCode  = getPhaseAxisCode(sim.signal.phase)
  const greenTypeCode     = getGreenTypeCode(sim.signal.phase)
  const neighborPhaseCode =
    typeof options.neighborPhase === 'number'
      ? options.neighborPhase
      : typeof sim.neighborPhase === 'number'
        ? sim.neighborPhase
        : currentPhaseCode
  return {
    state: {
      queue_pcu_NS:    ns.queuePcuBin,
      queue_pcu_EW:    ew.queuePcuBin,
      wait_NS:         ns.avgWaitBin,
      wait_EW:         ew.avgWaitBin,
      current_phase:   currentPhaseCode,
      neighbor_phase:  neighborPhaseCode,
      time_of_day_bin: getTimeOfDayBin(),
    },
    details: {
      queue_pcu_NS_raw:     ns.queuePcu,
      queue_pcu_EW_raw:     ew.queuePcu,
      wait_NS_raw_seconds:  ns.avgWait,
      wait_EW_raw_seconds:  ew.avgWait,
      current_green_type:   greenTypeCode,
      current_phase_name:   sim.signal.phase,
      pcu_standard:         PCU_BY_TYPE,
      bins: {
        queue_pcu:    '0:<5, 1:5-15, 2:>=15',
        wait_seconds: '0:<30, 1:30-90, 2:>=90',
      },
    },
  }
}

function getMlTrafficSnapshot(sim) {
  const laneCounts = {
    N2S: { lane0: 0, lane1: 0, total: 0 },
    S2N: { lane0: 0, lane1: 0, total: 0 },
    W2E: { lane0: 0, lane1: 0, total: 0 },
    E2W: { lane0: 0, lane1: 0, total: 0 },
  }
  for (const v of sim.vehicles) {
    laneCounts[v.direction].total++
    if (v.laneIndex === 0) laneCounts[v.direction].lane0++
    else laneCounts[v.direction].lane1++
  }
  const lights = sim.signal.lights ?? getFallbackLightsFromPhase(sim.signal.phase, sim.signal.phase_type)
  const arrowState = {
    R1: { left: !!lights.R1?.arrow_left, rs: !!lights.R1?.arrow_rs },
    R2: { left: !!lights.R2?.arrow_left, rs: !!lights.R2?.arrow_rs },
    R3: { left: !!lights.R3?.arrow_left, rs: !!lights.R3?.arrow_rs },
    R4: { left: !!lights.R4?.arrow_left, rs: !!lights.R4?.arrow_rs },
  }
  return {
    phase: sim.signal.phase,
    phase_type: sim.signal.phase_type ?? 'green',
    lane_counts_by_side: laneCounts,
    current_light_state: lights,
    current_arrow_state: arrowState,
    by_direction: {
      N2S: { pole: 'R1', lights: lights.R1, arrows: arrowState.R1 },
      S2N: { pole: 'R2', lights: lights.R2, arrows: arrowState.R2 },
      W2E: { pole: 'R3', lights: lights.R3, arrows: arrowState.R3 },
      E2W: { pole: 'R4', lights: lights.R4, arrows: arrowState.R4 },
    },
  }
}

function buildRLInferenceState(sim) {
  const clamp01 = (v) => Math.max(0, Math.min(1, v))
  let NS = 0, EW = 0, NSL = 0, EWL = 0
  for (const v of sim.vehicles) {
    const nearStop = v.distance < v.path.stopDistance + 44
    if (!nearStop) continue
    const isNS = axisForDirection(v.direction) === 'NS'
    if (isNS) {
      if (v.maneuver === 'left') NSL += 1
      else NS += 1
    } else {
      if (v.maneuver === 'left') EWL += 1
      else EW += 1
    }
  }

  const phaseId = Math.max(0, RL_PHASES.indexOf(sim.signal.phase))
  const elapsedSteps = Number.isFinite(sim.signal.elapsed) ? sim.signal.elapsed : 0

  return [
    clamp01(NS / STATE_QUEUE_NORM),
    clamp01(EW / STATE_QUEUE_NORM),
    clamp01(NSL / STATE_QUEUE_NORM),
    clamp01(EWL / STATE_QUEUE_NORM),
    phaseId / Math.max(1, RL_PHASES.length - 1),
    Math.min(elapsedSteps / MAX_GREEN_STEPS, 1.0),
  ]
}

function nextRLSignalState(signal, requestedAction, dt, state, frontDemand = null, liveQueues = null) {
  const elapsed = (signal.elapsed ?? 0) + dt

  if (signal.phase_type === 'all_red') {
    return { phase: signal.phase, phase_type: 'green', elapsed: 0 }
  }

  if (signal.phase_type === 'yellow') {
    return { phase: signal.phase, phase_type: 'green', elapsed: 0 }
  }

  const currentIndex = Math.max(0, RL_PHASES.indexOf(signal.phase))
  const currentPhase = RL_PHASES[currentIndex]
  const totalCongestion = (liveQueues?.NS ?? 0) + (liveQueues?.EW ?? 0)
  const inCongestionOverride = totalCongestion > RL_CONGESTION_CAP
  const currentFrontDemand = frontDemand?.[currentPhase] ?? 0
  const highestFrontPhase = bestPhaseByFrontDemand(frontDemand, currentPhase)
  const highestFrontDemand = frontDemand?.[highestFrontPhase] ?? 0

  if (inCongestionOverride) {
    const pressurePhase = bestPhaseByPressure(state, currentPhase)
    const emergencyPhase = highestFrontDemand > 0 ? highestFrontPhase : pressurePhase
    if (currentPhase !== emergencyPhase || totalCongestion > RL_CONGESTION_CAP + 2) {
      return { phase: emergencyPhase, phase_type: 'green', elapsed: 0 }
    }
    if (totalCongestion > RL_CONGESTION_RECOVERY) {
      return { phase: currentPhase, phase_type: 'green', elapsed }
    }
  }

  // If current green serves nobody near the stop line, redirect quickly.
  if (elapsed >= 2 && currentFrontDemand === 0 && highestFrontDemand > 0) {
    return { phase: highestFrontPhase, phase_type: 'green', elapsed: 0 }
  }

  const holdLocked = elapsed < MIN_GREEN_STEPS
  const forceSwitch = elapsed >= MAX_GREEN_STEPS
  let action = requestedAction === 1 ? 1 : 0
  if (holdLocked) action = 0
  if (forceSwitch) action = 1

  if (action === 1) {
    const activePressure = phasePressureFromState(state, currentPhase)
    const targetPhase = bestPhaseByPressure(state, currentPhase)
    const targetPressure = phasePressureFromState(state, targetPhase)
    if (!forceSwitch && activePressure > targetPressure + 0.03) {
      action = 0
    } else {
      return { phase: targetPhase, phase_type: 'green', elapsed: 0 }
    }
  }

  return { phase: currentPhase, phase_type: 'green', elapsed }
}

// ---------------------------------------------------------------------------
// Vehicle factory & movement helpers (unchanged)
// ---------------------------------------------------------------------------
function createVehicle(id, direction, vehicleMix) {
  const roll = Math.random()
  const maneuver = roll < 0.2 ? 'right' : roll < 0.35 ? 'left' : 'straight'
  const type = pickVehicleType(vehicleMix)
  const rightTurnLaneByDirection = { N2S: 0, S2N: 1, W2E: 1, E2W: 0 }
  const leftTurnLaneByDirection  = { N2S: 1, S2N: 0, W2E: 0, E2W: 1 }
  const laneIndex =
    maneuver === 'right'
      ? rightTurnLaneByDirection[direction]
      : maneuver === 'left'
        ? leftTurnLaneByDirection[direction]
        : Math.floor(Math.random() * 2)
  const path = buildPath(direction, maneuver, laneIndex)
  return {
    id, type, direction, maneuver, laneIndex, path,
    distance: 0,
    speed: BASE_SPEED * (0.85 + Math.random() * 0.35),
    spriteKey: type,
    joints: 0,
    wait: 0,
    stuckTime: 0,
    laneChange: null,
    laneChangeCooldown: 0,
  }
}

function canUseAlternateLane(vehicle, vehicles) {
  const altLane  = 1 - vehicle.laneIndex
  const safeBand = SAFE_GAP + getVehicleCollisionRadius(vehicle.type) * 2
  return !vehicles.some(
    (other) =>
      other.id !== vehicle.id &&
      other.direction === vehicle.direction &&
      other.laneIndex === altLane &&
      Math.abs(other.distance - vehicle.distance) < safeBand,
  )
}

function tryLaneChangeForStraight(vehicle, vehicles, signal) {
  if (vehicle.maneuver !== 'straight') return
  if (vehicle.laneChange || vehicle.laneChangeCooldown > 0) return
  const stage = parseSignalPhase(signal.phase)
  if (stage.axis !== axisForDirection(vehicle.direction) || stage.movement !== 'straight') return
  if (vehicle.distance >= vehicle.path.stopDistance - 10) return

  const blocker = vehicles
    .filter(
      (other) =>
        other.id !== vehicle.id &&
        other.direction === vehicle.direction &&
        other.laneIndex === vehicle.laneIndex &&
        other.distance > vehicle.distance,
    )
    .sort((a, b) => a.distance - b.distance)[0]

  if (!blocker) return
  const blockerCannotGoNow  = !canVehicleProceedFromBackend(signal, blocker)
  const blockerNearStop     = blocker.distance < blocker.path.stopDistance + 16
  const blockerIsTurnConflict = blocker.maneuver === 'left' || blockerCannotGoNow
  const closeEnoughToBeImpacted = blocker.distance - vehicle.distance < SAFE_GAP + 38
  if (!blockerIsTurnConflict || !blockerNearStop || !closeEnoughToBeImpacted) return
  if (!canUseAlternateLane(vehicle, vehicles)) return

  const altLane = 1 - vehicle.laneIndex
  const toPath  = buildPath(vehicle.direction, vehicle.maneuver, altLane)
  vehicle.laneChange = {
    fromPath: vehicle.path,
    progress: 0,
    duration: 0.55 + Math.random() * 0.28,
    sway:     (Math.random() - 0.5) * 1.6,
  }
  vehicle.laneIndex = altLane
  vehicle.path      = toPath
  vehicle.laneChangeCooldown = 2.1 + Math.random() * 0.7
}

function sampleVehiclePose(vehicle, distance) {
  if (!vehicle.laneChange) return samplePath(vehicle.path, distance)
  const fromPose = samplePath(vehicle.laneChange.fromPath, distance)
  const toPose   = samplePath(vehicle.path, distance)
  const t        = easeInOutSine(Math.max(0, Math.min(1, vehicle.laneChange.progress)))
  const x        = lerp(fromPose.x, toPose.x, t)
  const y        = lerp(fromPose.y, toPose.y, t)
  const sinMix   = (1 - t) * Math.sin(fromPose.angle) + t * Math.sin(toPose.angle)
  const cosMix   = (1 - t) * Math.cos(fromPose.angle) + t * Math.cos(toPose.angle)
  const angle    = Math.atan2(sinMix, cosMix)
  const sway     = Math.sin(t * Math.PI) * vehicle.laneChange.sway
  const normalAngle = angle + Math.PI / 2
  return { x: x + Math.cos(normalAngle) * sway, y: y + Math.sin(normalAngle) * sway, angle }
}

function getVehicleBounds(v, distance) {
  const p = sampleVehiclePose(v, distance)
  const r = Math.max(VEHICLE_SOLID_RADIUS, getVehicleCollisionRadius(v.type))
  return { x: p.x, y: p.y, r }
}

function intersectsVehicleAtDistance(vehicle, targetDistance, vehicles) {
  const mine = getVehicleBounds(vehicle, targetDistance)
  for (const other of vehicles) {
    if (other.id === vehicle.id) continue
    if (other.direction !== vehicle.direction) continue
    if (other.laneIndex !== vehicle.laneIndex) continue
    const otherPos = getVehicleBounds(other, other.distance)
    const minGap   = mine.r + otherPos.r + 5
    if (Math.hypot(mine.x - otherPos.x, mine.y - otherPos.y) < minGap) return true
  }
  return false
}

// ---------------------------------------------------------------------------
// Canvas drawing helpers
// ---------------------------------------------------------------------------
function drawRoundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function drawCarSprite(ctx, image, alpha, isLeadSegment, vehicleType) {
  const dims   = getVehicleRenderDims(vehicleType, isLeadSegment)
  const length = dims.length, width = dims.width
  ctx.fillStyle = `rgba(0,0,0,${0.2 * alpha})`
  ctx.beginPath()
  ctx.ellipse(0, 0.7, length * 0.47, width * 0.4, 0, 0, Math.PI * 2)
  ctx.fill()
  if (!image || !image.complete) {
    ctx.fillStyle = `rgba(230,230,230,${0.82 * alpha})`
    drawRoundedRect(ctx, -length / 2, -width / 2, length, width, 4)
    ctx.fill()
    return
  }
  ctx.globalAlpha = alpha
  ctx.drawImage(image, -length / 2, -width / 2, length, width)
  ctx.globalAlpha = 1
}

function queueSeverityColor(value, max = 120) {
  const t = Math.max(0, Math.min(1, value / max))
  if (t < 0.35) return '#4ddf90'
  if (t < 0.65) return '#ffca5f'
  return '#ff6f59'
}

function phasePressureFromState(state, phase) {
  if (!state || state.length < 4) return 0
  const idx = RL_STATE_INDEX[phase]
  return idx == null ? 0 : (state[idx] ?? 0)
}

function bestPhaseByPressure(state, currentPhase) {
  if (!state || state.length < 4) return currentPhase
  let best = currentPhase
  let bestPressure = phasePressureFromState(state, currentPhase)
  for (const phase of RL_PHASES) {
    const p = phasePressureFromState(state, phase)
    if (p > bestPressure) {
      bestPressure = p
      best = phase
    }
  }
  return best
}

function bestPhaseByFrontDemand(frontDemand, currentPhase) {
  if (!frontDemand) return currentPhase
  let best = currentPhase
  let bestDemand = frontDemand[currentPhase] ?? 0
  for (const phase of RL_PHASES) {
    const d = frontDemand[phase] ?? 0
    if (d > bestDemand) {
      bestDemand = d
      best = phase
    }
  }
  return best
}

function drawRoadHeatMap(ctx, queues) {
  const nsColor = queueSeverityColor(queues.NS, 160)
  const ewColor = queueSeverityColor(queues.EW, 160)
  const nsAlpha = Math.max(0.08, Math.min(0.42, queues.NS / 280))
  const ewAlpha = Math.max(0.08, Math.min(0.42, queues.EW / 280))

  const nsRgb = nsColor === '#4ddf90' ? '77,223,144' : nsColor === '#ffca5f' ? '255,202,95' : '255,111,89'
  const ewRgb = ewColor === '#4ddf90' ? '77,223,144' : ewColor === '#ffca5f' ? '255,202,95' : '255,111,89'

  ctx.fillStyle = `rgba(${nsRgb}, ${nsAlpha})`
  ctx.fillRect(-WORLD.road / 2, -WORLD.height / 2, WORLD.road, WORLD.height)
  ctx.fillStyle = `rgba(${ewRgb}, ${ewAlpha})`
  ctx.fillRect(-WORLD.width / 2, -WORLD.road / 2, WORLD.width, WORLD.road)
}

function drawQueueGauge(ctx, { label, value, x, y, width = 190, height = 18 }) {
  const ratio = Math.max(0, Math.min(1, value / 120))
  const color = queueSeverityColor(value, 120)

  ctx.fillStyle = 'rgba(12,22,29,0.82)'
  drawRoundedRect(ctx, x, y, width, 56, 10)
  ctx.fill()

  ctx.fillStyle = '#dbe7ed'
  ctx.font = '700 12px "Space Grotesk", sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x + 10, y + 15)

  ctx.fillStyle = '#1f2d36'
  drawRoundedRect(ctx, x + 10, y + 24, width - 20, height, 7)
  ctx.fill()

  ctx.fillStyle = color
  drawRoundedRect(ctx, x + 10, y + 24, (width - 20) * ratio, height, 7)
  ctx.fill()

  ctx.fillStyle = '#f7f9fb'
  ctx.font = '700 13px "Space Grotesk", sans-serif'
  ctx.textAlign = 'right'
  ctx.fillText(`${Math.round(value)} queued`, x + width - 10, y + 44)
}

function drawVisualDeltaHUD(ctx, { queues, mode, benchmarkGain }) {
  const total = queues.NS + queues.EW
  drawQueueGauge(ctx, { label: 'NS Queue Pressure', value: queues.NS, x: -462, y: -274 })
  drawQueueGauge(ctx, { label: 'EW Queue Pressure', value: queues.EW, x: 272, y: -274 })

  ctx.fillStyle = 'rgba(14, 23, 33, 0.88)'
  drawRoundedRect(ctx, -112, -286, 224, 72, 12)
  ctx.fill()

  ctx.fillStyle = queueSeverityColor(total, 180)
  ctx.font = '800 18px "Space Grotesk", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(`Live Congestion: ${Math.round(total)}`, 0, -260)

  ctx.fillStyle = '#d7e4ea'
  ctx.font = '700 12px "Space Grotesk", sans-serif'
  const gainText = Number.isFinite(benchmarkGain)
    ? `Benchmark uplift vs fixed: ${benchmarkGain.toFixed(1)}%`
    : 'Run benchmark to show uplift vs fixed'
  ctx.fillText(`${mode.toUpperCase()} MODE | ${gainText}`, 0, -238)
}

// Returns {red, yellow, green} for a single signal head, respecting phase_type
function getSignalHeadState(phase, phaseType, axis) {
  if (phaseType === 'all_red') return { red: true, yellow: false, green: false }
  if (phaseType === 'yellow') {
    const stage    = parseSignalPhase(phase)
    const isActive = stage.axis === axis
    return { red: !isActive, yellow: isActive, green: false }
  }
  // green
  const stage    = parseSignalPhase(phase)
  const isActive = stage.axis === axis
  return { red: !isActive, yellow: false, green: isActive }
}

function drawApproachSignal(ctx, config, phase, phaseType) {
  const stage  = parseSignalPhase(phase)
  const active = stage.axis === config.axis
  const state  = getSignalHeadState(phase, phaseType, config.axis)

  ctx.save()
  ctx.translate(config.x, config.y)
  ctx.strokeStyle = '#5a6068'
  ctx.lineWidth   = 4
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.lineTo(config.armX, config.armY)
  ctx.stroke()
  ctx.fillStyle = '#67707a'
  drawRoundedRect(ctx, -5, -12, 10, 24, 3)
  ctx.fill()
  ctx.translate(config.armX, config.armY)

  ctx.fillStyle = '#0f1820f0'
  drawRoundedRect(ctx, -34, -68, 68, 136, 16)
  ctx.fill()

  const borderColor = state.green ? '#45d26e' : state.yellow ? '#ffc857' : '#ff5a5a'
  ctx.strokeStyle = borderColor
  ctx.lineWidth   = 2.25
  drawRoundedRect(ctx, -34, -68, 68, 136, 16)
  ctx.stroke()

  ctx.fillStyle   = active ? '#102b1a' : '#202b31'
  drawRoundedRect(ctx, -18, -62, 36, 16, 6)
  ctx.fill()
  ctx.strokeStyle = active ? '#67e28a' : '#90a0a8'
  ctx.lineWidth   = 1.4
  drawRoundedRect(ctx, -18, -62, 36, 16, 6)
  ctx.stroke()
  ctx.fillStyle     = active ? '#d8ffe0' : '#d9e2e6'
  ctx.font          = '800 9px "Space Grotesk", sans-serif'
  ctx.textAlign     = 'center'
  ctx.textBaseline  = 'middle'
  ctx.fillText(`R${config.roadNumber ?? 0}`, 0, -54)

  const bulbs = [
    { x: 0, y: -28, color: state.red    ? '#ff5d5d' : '#44525c' },
    { x: 0, y:   0, color: state.yellow ? '#ffc857' : '#44525c' },
    { x: 0, y:  28, color: state.green  ? '#4bdb72' : '#44525c' },
  ]
  for (const bulb of bulbs) {
    ctx.fillStyle = bulb.color
    ctx.beginPath()
    ctx.arc(bulb.x, bulb.y, 9.2, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()
}

function getApproachIcon(direction, movement) {
  if (movement === 'straight')
    return { N2S: FaArrowDown, S2N: FaArrowUp, W2E: FaArrowRight, E2W: FaArrowLeft }[direction]
  if (movement === 'right')
    return { N2S: FaArrowLeft, S2N: FaArrowRight, W2E: FaArrowDown, E2W: FaArrowUp }[direction]
  return { N2S: FaArrowRight, S2N: FaArrowLeft, W2E: FaArrowUp, E2W: FaArrowDown }[direction]
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
function App() {
  const canvasRef      = useRef(null)
  const carImagesRef   = useRef({})
  const rafRef         = useRef(0)
  const lastFrameRef   = useRef(0)
  const seqRef         = useRef(1)
  const signalFetchRef = useRef(0)
  const failureUntilRef = useRef(0)

  const simRef = useRef({
    vehicles: [],
    signal: { phase: 'NS_straight', phase_type: 'green', elapsed: 0, lights: null },
    spawnClock: { N2S: 0, S2N: 1, W2E: 0.5, E2W: 1.4 },
    completed: 0,
    totalWait: 0,
    simTime: 0,
  })

  const [running,  setRunning]  = useState(true)
  const [density,  setDensity]  = useState(58)
  const [mode, setMode] = useState('fixed')
  const [vehicleMix, setVehicleMix] = useState('kathmandu')
  const [simSpeed, setSimSpeed] = useState(1)
  const [visualOverlay, setVisualOverlay] = useState(true)
  const [rlDecisionSource, setRlDecisionSource] = useState('frontend')
  const [trainingPreset, setTrainingPreset] = useState('balanced')
  const [trainingSeed, setTrainingSeed] = useState(12345)
  const [validationStatus, setValidationStatus] = useState('idle')
  const [backendModelStatus, setBackendModelStatus] = useState('not-loaded')
  const [view, setView] = useState({
    phase: 'NS_straight', phase_type: 'green',
    queues: { NS: 0, EW: 0 },
    completed: 0, avgWait: 0, efficiency: 0, liveCars: 0, simTime: 0,
  })

  const backendStrategy = 'fixed'

  const { startTraining, abortTraining, trainingState, progress, result, error } = useRLTrainer()
  const rlController = useRLController()
  const [validationResult, setValidationResult] = useState(null)

  // Auto-load saved model on mount
  useEffect(() => { rlController.loadFromStorage() }, [])

  // Auto-validate when training finishes
  useEffect(() => {
    if (result?.weights) {
      rlController.loadModel(result.weights)
      setValidationStatus('running')
      runValidation(result.weights)
        .then((res) => {
          setValidationResult(res)
          setValidationStatus('done')
        })
        .catch(() => setValidationStatus('error'))
    }
  }, [result])

  async function runBenchmark() {
    if (!rlController.loadedWeights) return
    setValidationStatus('running')
    try {
      const res = await runValidation(rlController.loadedWeights)
      setValidationResult(res)
      setValidationStatus('done')
    } catch {
      setValidationStatus('error')
    }
  }

  async function pushModelToBackend() {
    if (!rlController.loadedWeights) return
    setBackendModelStatus('uploading')
    try {
      const r = await fetch('http://127.0.0.1:8000/rl/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rlController.loadedWeights),
      })
      if (!r.ok) throw new Error('upload failed')
      setBackendModelStatus('loaded')
    } catch {
      setBackendModelStatus('error')
    }
  }

  const modeLabel = useMemo(() => {
    if (mode === 'rl') return 'RL (Trained Model)'
    return 'Fixed'
  }, [mode])

  const approachSignals = useMemo(() => [
    { axis: 'NS', direction: 'N2S', label: 'N->S' },
    { axis: 'NS', direction: 'S2N', label: 'S->N' },
    { axis: 'EW', direction: 'W2E', label: 'W->E' },
    { axis: 'EW', direction: 'E2W', label: 'E->W' },
  ], [])

  const signalHeads = useMemo(() => [
    { ...approachSignals[0], roadNumber: 1, x: -244, y: -190, armX:  56, armY:   0 },
    { ...approachSignals[1], roadNumber: 2, x:  244, y:  190, armX: -56, armY:   0 },
    { ...approachSignals[2], roadNumber: 3, x: -190, y:  244, armX:   0, armY: -56 },
    { ...approachSignals[3], roadNumber: 4, x:  190, y: -244, armX:   0, armY:  56 },
  ], [approachSignals])

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')

    for (const [key, src] of Object.entries(VEHICLE_SPRITES)) {
      const img = new Image()
      img.src = src
      carImagesRef.current[key] = img
    }

    function renderWorld(state) {
      const liveQueues = countQueues(state.vehicles, state.signal)
      const benchmarkGain = validationResult
        ? Number(improvementPercent(validationResult.fixed.summary.avgQueue, validationResult.rl.summary.avgQueue, 'queue').replace('%', ''))
        : null

      ctx.clearRect(0, 0, WORLD.width, WORLD.height)
      const grad = ctx.createLinearGradient(0, 0, WORLD.width, WORLD.height)
      grad.addColorStop(0, '#f3efe2')
      grad.addColorStop(1, '#dcece5')
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, WORLD.width, WORLD.height)

      ctx.save()
      ctx.translate(WORLD.width / 2, WORLD.height / 2)

      const halfRoad = WORLD.road / 2
      ctx.fillStyle = '#262d35'
      ctx.fillRect(-WORLD.width / 2, -halfRoad, WORLD.width, WORLD.road)
      ctx.fillRect(-halfRoad, -WORLD.height / 2, WORLD.road, WORLD.height)
      if (visualOverlay) drawRoadHeatMap(ctx, liveQueues)

      ctx.strokeStyle = '#f4c430'
      ctx.lineWidth   = 3
      ctx.beginPath()
      ctx.moveTo(-WORLD.width / 2, 0); ctx.lineTo(WORLD.width / 2, 0)
      ctx.moveTo(0, -WORLD.height / 2); ctx.lineTo(0, WORLD.height / 2)
      ctx.stroke()

      ctx.strokeStyle = '#f7f7ee'
      ctx.lineWidth   = 2
      ctx.setLineDash([14, 10])
      ctx.beginPath()
      ctx.moveTo(-WORLD.width / 2, -WORLD.laneOffset); ctx.lineTo(WORLD.width / 2, -WORLD.laneOffset)
      ctx.moveTo(-WORLD.width / 2,  WORLD.laneOffset); ctx.lineTo(WORLD.width / 2,  WORLD.laneOffset)
      ctx.moveTo(-WORLD.laneOffset, -WORLD.height / 2); ctx.lineTo(-WORLD.laneOffset, WORLD.height / 2)
      ctx.moveTo( WORLD.laneOffset, -WORLD.height / 2); ctx.lineTo( WORLD.laneOffset, WORLD.height / 2)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.strokeStyle = '#efefe4'
      ctx.lineWidth   = 4
      ctx.beginPath()
      ctx.moveTo(-halfRoad + 10, -halfRoad + 2); ctx.lineTo(halfRoad - 10, -halfRoad + 2)
      ctx.moveTo(-halfRoad + 10,  halfRoad - 2); ctx.lineTo(halfRoad - 10,  halfRoad - 2)
      ctx.moveTo(-halfRoad + 2, -halfRoad + 10); ctx.lineTo(-halfRoad + 2,  halfRoad - 10)
      ctx.moveTo( halfRoad - 2, -halfRoad + 10); ctx.lineTo( halfRoad - 2,  halfRoad - 10)
      ctx.stroke()

      ctx.fillStyle    = '#f8f7f0'
      ctx.font         = '800 18px "Space Grotesk", sans-serif'
      ctx.textAlign    = 'center'
      ctx.textBaseline = 'middle'
      for (const label of [
        { text: '1', x: -12, y: -118 }, { text: '2', x:  12, y: 118 },
        { text: '3', x: -118, y: 12  }, { text: '4', x: 118, y: -12 },
      ]) ctx.fillText(label.text, label.x, label.y)

      // Draw signal heads — pass phase_type so yellow/all-red render correctly
      const phaseType = state.signal.phase_type ?? 'green'
      for (const config of signalHeads) {
        drawApproachSignal(ctx, config, state.signal.phase, phaseType)
      }

      for (const v of state.vehicles) {
        for (let j = v.joints; j >= 0; j--) {
          const segDist = v.distance - j * JOINT_GAP
          const p       = sampleVehiclePose(v, segDist)
          const alpha   = j === 0 ? 1 : 0.78
          const sprite  = carImagesRef.current[v.spriteKey]
          ctx.save()
          ctx.translate(p.x, p.y)
          ctx.rotate(p.angle + (VEHICLE_HEADING_OFFSETS[v.type] ?? 0))

          if (visualOverlay && j === 0 && v.wait > 2.5) {
            const stress = Math.min(1, (v.wait - 2.5) / 8)
            const glow = 18 + stress * 20
            const glowColor = stress > 0.65 ? '255,90,90' : stress > 0.35 ? '255,196,87' : '120,255,170'
            ctx.fillStyle = `rgba(${glowColor}, ${0.16 + stress * 0.25})`
            ctx.beginPath()
            ctx.ellipse(0, 0, glow, glow * 0.56, 0, 0, Math.PI * 2)
            ctx.fill()
          }

          drawCarSprite(ctx, sprite, alpha, j === 0, v.type)
          ctx.restore()
          if (j > 0) {
            const p2 = sampleVehiclePose(v, v.distance - (j - 1) * JOINT_GAP)
            ctx.strokeStyle = '#111b'
            ctx.lineWidth   = 2
            ctx.beginPath()
            ctx.moveTo(p.x, p.y)
            ctx.lineTo(p2.x, p2.y)
            ctx.stroke()
          }
        }
      }

      if (visualOverlay) {
        drawVisualDeltaHUD(ctx, {
          queues: liveQueues,
          mode,
          benchmarkGain,
        })
      }

      ctx.restore()
    }

    function step(ts) {
      const prev = lastFrameRef.current || ts
      const dt   = Math.min(0.045, (ts - prev) / 1000)
      lastFrameRef.current = ts

      const sim = simRef.current
      if (running) {
        sim.simTime += dt*simSpeed

        const queues = countQueues(sim.vehicles, sim.signal)
        const approaching = {
          NS: countApproaching(sim.vehicles, 'NS'),
          EW: countApproaching(sim.vehicles, 'EW'),
        }

        // ── Signal tick (every 250 ms, scaled by simSpeed) ────────────────
       const SIGNAL_REAL_INTERVAL = 0.25
const signalDt = SIGNAL_REAL_INTERVAL * simSpeed
signalFetchRef.current += dt
if (signalFetchRef.current >= SIGNAL_REAL_INTERVAL) {
  signalFetchRef.current = 0

  const fallback = nextSignalLocal(sim.signal, backendStrategy, queues, approaching, signalDt)

          const backendMode = mode === 'rl' && rlDecisionSource === 'backend' && backendModelStatus === 'loaded'
          const payload = {
            current_phase: sim.signal.phase,
            phase_type:    sim.signal.phase_type ?? 'green',
            elapsed:       sim.signal.elapsed,
            dt:            signalDt,
            strategy: backendMode ? 'rl' : backendStrategy,
            queues,
            approaching,
            ml_snapshot:   getMlTrafficSnapshot(sim),
            ml_state:      buildSignalMlState(sim),
            rl_state:      backendMode ? buildRLInferenceState(sim) : null,
          }

          console.log('[SignalTick][Request]', {
            ts: new Date().toISOString(),
            phase: sim.signal.phase, phase_type: sim.signal.phase_type,
            elapsed: sim.signal.elapsed, strategy: backendStrategy, mode, queues, approaching,
            simSpeed,
          })

          const failureActive = sim.simTime < failureUntilRef.current
          if (failureActive) {
            console.log('[SignalTick][FailureMode]', {
              ts: new Date().toISOString(),
              until: failureUntilRef.current,
              phase: fallback.phase,
              phase_type: fallback.phase_type,
            })
            sim.signal.phase = fallback.phase
            sim.signal.phase_type = fallback.phase_type
            sim.signal.elapsed = fallback.elapsed
            sim.signal.lights = getFallbackLightsFromPhase(fallback.phase, fallback.phase_type)
          } else if (mode === 'rl' && rlDecisionSource === 'frontend' && rlController.isReady) {
            const state = buildRLInferenceState(sim)
            const phaseFrontDemand = getPhaseFrontDemand(sim.vehicles)
            const decision = rlController.getNextAction(state)
            const modelAction = decision?.action ?? 0

            const next = nextRLSignalState(sim.signal, modelAction, signalDt, state, phaseFrontDemand, queues)
            sim.signal.phase = next.phase
            sim.signal.phase_type = next.phase_type
            sim.signal.elapsed = next.elapsed
            sim.signal.lights = getFallbackLightsFromPhase(next.phase, next.phase_type)
          } else {
            fetch('http://127.0.0.1:8000/signal/tick', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            })
              .then((r) => r.json())
              .then((data) => {
                console.log('[SignalTick][Response]', {
                  ts: new Date().toISOString(),
                  phase: data.phase, phase_type: data.phase_type,
                  elapsed: data.elapsed, lights: data.lights,
                })
                sim.signal.phase      = data.phase
                sim.signal.phase_type = data.phase_type
                sim.signal.elapsed    = data.elapsed
                sim.signal.lights     = data.lights
              })
              .catch(() => {
                console.log('[SignalTick][Fallback]', {
                  ts: new Date().toISOString(),
                  phase: fallback.phase, phase_type: fallback.phase_type,
                  elapsed: fallback.elapsed,
                })
                sim.signal.phase      = fallback.phase
                sim.signal.phase_type = fallback.phase_type
                sim.signal.elapsed    = fallback.elapsed
                sim.signal.lights     = getFallbackLightsFromPhase(fallback.phase, fallback.phase_type)
              })
          }
        }

        // ── Spawn ─────────────────────────────────────────────────────────
        const simSteps = Math.max(1, Math.floor(simSpeed))
        const stepDt = dt
        for (let stepIndex = 0; stepIndex < simSteps; stepIndex += 1) {
          const minSpawn = lerp(3.1, 0.75, density / 100)
          for (const dir of DIRECTIONS) {
            sim.spawnClock[dir] -= stepDt
            if (sim.spawnClock[dir] <= 0) {
              const blocked = sim.vehicles.some((v) => v.direction === dir && v.distance < 96)
              if (!blocked) {
                sim.vehicles.push(createVehicle(seqRef.current, dir, vehicleMix))
                seqRef.current++
              }
              sim.spawnClock[dir] = minSpawn * (0.55 + Math.random() * 0.95)
            }
          }

        // ── Leader map ────────────────────────────────────────────────────
          const byLane = new Map()
          for (const v of sim.vehicles) {
            const key = `${v.direction}:${v.laneIndex}`
            if (!byLane.has(key)) byLane.set(key, [])
            byLane.get(key).push(v)
          }
          for (const lv of byLane.values()) lv.sort((a, b) => b.distance - a.distance)

          const leaderMap = new Map()
          for (const lv of byLane.values())
            for (let i = 1; i < lv.length; i++) leaderMap.set(lv[i].id, lv[i - 1])

        // ── Move vehicles ─────────────────────────────────────────────────
          for (const v of sim.vehicles) {
            if (v.laneChangeCooldown > 0) v.laneChangeCooldown = Math.max(0, v.laneChangeCooldown - stepDt)
          tryLaneChangeForStraight(v, sim.vehicles, sim.signal)

            let targetDistance = v.distance + v.speed * stepDt

          const allowed = canVehicleProceedFromBackend(sim.signal, v)
          if (!allowed && v.distance < v.path.stopDistance)
            targetDistance = Math.min(targetDistance, v.path.stopDistance - 4)

          const leader = leaderMap.get(v.id)
          if (leader) {
            const extraTurnGap = v.maneuver !== 'straight' || leader.maneuver !== 'straight' ? 10 : 0
            targetDistance = Math.min(targetDistance, leader.distance - (getFollowingGap(v, leader) + extraTurnGap))
          }

          if (intersectsVehicleAtDistance(v, targetDistance, sim.vehicles))
            targetDistance = v.distance

          targetDistance = Math.max(v.distance, targetDistance)
          const moved = targetDistance - v.distance
            if (moved < 0.4) { v.wait += stepDt; sim.totalWait += stepDt }

            const blockedByRed = !allowed && v.distance < v.path.stopDistance - 2
            v.stuckTime = moved < 0.2 && !blockedByRed ? (v.stuckTime ?? 0) + stepDt : 0
            v.distance  = targetDistance

            if ((v.stuckTime ?? 0) > STUCK_RECOVERY_SECONDS) {
            const unstickDist = Math.min(v.distance + STUCK_NUDGE_DISTANCE, v.path.length)
            if (!intersectsVehicleAtDistance(v, unstickDist, sim.vehicles)) {
              v.distance = unstickDist; v.stuckTime = 0
            } else if ((v.stuckTime ?? 0) > STUCK_FAILSAFE_SECONDS) {
              v.distance = v.path.length + VEHICLE_LENGTH * 2; v.stuckTime = 0
            }
            }

            if (v.laneChange) {
              v.laneChange.progress += stepDt / v.laneChange.duration
              if (v.laneChange.progress >= 1) v.laneChange = null
            }
          }

          sim.vehicles = sim.vehicles.filter((v) => {
            const keep = v.distance < v.path.length + VEHICLE_LENGTH * 2
            if (!keep) sim.completed++
            return keep
          })
        }

        if (Math.floor(sim.simTime * 4) !== Math.floor((sim.simTime - dt) * 4)) {
          const latestQueues = countQueues(sim.vehicles, sim.signal)
          setView({
            phase:      sim.signal.phase,
            phase_type: sim.signal.phase_type ?? 'green',
            queues:     latestQueues,
            completed:  sim.completed,
            avgWait:    sim.completed === 0 ? 0 : sim.totalWait / sim.completed,
            efficiency: sim.simTime > 0 ? (sim.completed / sim.simTime) * 60 : 0,
            liveCars:   sim.vehicles.length,
            simTime:    sim.simTime,
          })
        }
      }

      renderWorld(simRef.current)
      rafRef.current = requestAnimationFrame(step)
    }

    rafRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(rafRef.current)
  }, [approachSignals, backendStrategy, density, mode, rlDecisionSource, running, signalHeads, simSpeed, validationResult, vehicleMix, visualOverlay])

  function resetSimulation() {
    simRef.current = {
      vehicles: [],
      signal: { phase: 'NS_straight', phase_type: 'green', elapsed: 0, lights: null },
      spawnClock: { N2S: 0, S2N: 1, W2E: 0.5, E2W: 1.4 },
      completed: 0, totalWait: 0, simTime: 0,
    }
    seqRef.current = 1
    setView({
      phase: 'NS_straight', phase_type: 'green',
      queues: { NS: 0, EW: 0 },
      completed: 0, avgWait: 0, efficiency: 0, liveCars: 0, simTime: 0,
    })
  }

  const phaseLabel =
    view.phase_type === 'all_red' ? 'ALL RED'
    : view.phase_type === 'yellow' ? `${view.phase.replace('_', ' ')} — YELLOW`
    : view.phase.replace('_', ' ')

  return (
    <div className="app-shell">
      <header className="hero-panel">
        <p className="eyebrow">Traffic Optimization System</p>
        <h1>Frontend Traffic Control Simulator</h1>
        <p className="subtitle">
          Multi-joint articulated vehicles, adaptive traffic lights, and live optimization metrics in a pure React application.
        </p>
      </header>

      <main className="layout-grid">
        <section className="card controls" aria-label="Simulation Controls">
          <h2>Control Center</h2>

          <label>Mode</label>
          <div className="toggle-row" role="group" aria-label="Simulation mode">
            <button type="button" className={mode === 'fixed' ? 'toggle-btn active' : 'toggle-btn'} onClick={() => setMode('fixed')}>
              Fixed
            </button>
            <button type="button" className={mode === 'rl' ? 'toggle-btn active' : 'toggle-btn'} onClick={() => setMode('rl')}>
              RL
            </button>
          </div>

          <label htmlFor="density">Traffic Density ({density}%)</label>
          <input
            id="density" type="range" min="10" max="100" value={density}
            onChange={(e) => setDensity(Number(e.target.value))}
          />

          <label>Vehicle Mix</label>
          <div className="toggle-row" role="group" aria-label="Vehicle mix">
            <button
              type="button"
              className={vehicleMix === 'kathmandu' ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setVehicleMix('kathmandu')}
            >
              Kathmandu
            </button>
            <button
              type="button"
              className={vehicleMix === 'car_heavy' ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setVehicleMix('car_heavy')}
            >
              Car-Heavy
            </button>
          </div>

          <label>Simulation Speed</label>
          <div className="toggle-row speed" role="group" aria-label="Simulation speed">
            {[1, 10, 100].map((s) => (
              <button
                key={s}
                type="button"
                className={simSpeed === s ? 'toggle-btn active' : 'toggle-btn'}
                onClick={() => setSimSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>

          <label>Visual Difference Overlay</label>
          <div className="toggle-row" role="group" aria-label="Visual difference overlay">
            <button
              type="button"
              className={visualOverlay ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setVisualOverlay(true)}
            >
              On
            </button>
            <button
              type="button"
              className={!visualOverlay ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setVisualOverlay(false)}
            >
              Off
            </button>
          </div>

          <label>Events</label>
          <div className="event-row">
            <button
              type="button"
              className="event-btn danger"
              onClick={() => {
                failureUntilRef.current = simRef.current.simTime + 12
                console.log('[Event] Triggered signal failure mode for 12s')
              }}
            >
              Trigger Signal Failure
            </button>
            <button type="button" className="event-btn" onClick={resetSimulation}>
              Reset Simulation
            </button>
          </div>

          <div className="button-row">
            <button type="button" onClick={() => setRunning((v) => !v)}>
              {running ? 'Pause' : 'Resume'}
            </button>
            <button type="button" className="secondary" onClick={() => setRunning(true)}>
              Resume Live
            </button>
          </div>

          {mode === 'rl' && (
            <div className="rl-panel">
              <p className="rl-status">Training status: <strong>{trainingState}</strong></p>
              {error && <p className="rl-error">Training error: {error}</p>}
              <label htmlFor="rl-preset">Training Preset</label>
              <select id="rl-preset" value={trainingPreset} onChange={(e) => setTrainingPreset(e.target.value)}>
                {Object.entries(TRAINING_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>{preset.label}</option>
                ))}
              </select>
              <label htmlFor="rl-seed">Training Seed</label>
              <input
                id="rl-seed"
                type="number"
                min="1"
                step="1"
                value={trainingSeed}
                onChange={(e) => setTrainingSeed(Math.max(1, Number(e.target.value) || 1))}
              />
              {trainingState === 'idle' && (
                <button
                  type="button"
                  onClick={() => startTraining({ ...TRAINING_PRESETS[trainingPreset], seed: trainingSeed, presetName: trainingPreset })}
                >
                  Start Training
                </button>
              )}
              {trainingState === 'training' && (
                <>
                  <div className="progress-bar">
                    <div style={{ width: `${progress.percentComplete}%` }} />
                  </div>
                  <p>Episode {progress.ep} / {progress.totalEpisodes} — ε {progress.epsilon}</p>
                  <p>Reward: {progress.episodeReward}  Queue: {progress.avgQueue}</p>
                  <button type="button" onClick={abortTraining}>Stop Training</button>
                </>
              )}
              {trainingState === 'done' && (
                <>
                  <p>Training complete — {result.finalEpisode} episodes</p>
                  <p>
                    Preset: <strong>{result.runInfo?.config?.presetName ?? trainingPreset}</strong> |
                    Seed: <strong>{result.runInfo?.config?.seed ?? trainingSeed}</strong>
                  </p>
                  <p>
                    Runtime: <strong>{((result.runInfo?.durationMs ?? 0) / 1000).toFixed(1)}s</strong> |
                    Train steps: <strong>{result.runInfo?.totalTrainSteps?.toLocaleString() ?? 'n/a'}</strong>
                  </p>
                  <button
                    type="button"
                    onClick={() => startTraining({ ...TRAINING_PRESETS[trainingPreset], seed: trainingSeed, presetName: trainingPreset })}
                  >
                    Train Again
                  </button>
                  <button type="button" onClick={() => exportToFile(result.weights)}>
                    Download Model
                  </button>
                </>
              )}
              <label>
                Load model
                <input type="file" accept=".json" onChange={async (e) => {
                  const obj = await importFromFile(e.target.files[0])
                  rlController.loadModel(obj)
                }} />
              </label>
              {rlController.isReady && (
                <>
                  <p>
                    Model ready — {rlController.modelMeta?.totalSteps?.toLocaleString()} steps |
                    version {rlController.modelMeta?.version ?? 1}
                  </p>
                  <div className="rl-row-buttons">
                    <button type="button" onClick={runBenchmark}>Run Benchmark</button>
                    <button type="button" onClick={pushModelToBackend}>Upload Model To Backend</button>
                    <button
                      type="button"
                      onClick={() => {
                        clearModel()
                        window.location.reload()
                      }}
                    >
                      Clear Saved Model
                    </button>
                  </div>
                </>
              )}
              <label>RL Inference Source</label>
              <div className="toggle-row" role="group" aria-label="RL inference source">
                <button
                  type="button"
                  className={rlDecisionSource === 'frontend' ? 'toggle-btn active' : 'toggle-btn'}
                  onClick={() => setRlDecisionSource('frontend')}
                >
                  Frontend
                </button>
                <button
                  type="button"
                  className={rlDecisionSource === 'backend' ? 'toggle-btn active' : 'toggle-btn'}
                  onClick={() => setRlDecisionSource('backend')}
                  disabled={!rlController.isReady}
                >
                  Backend
                </button>
              </div>
              <p>Backend model: <strong>{backendModelStatus}</strong></p>
              <p>Benchmark status: <strong>{validationStatus}</strong></p>
              {validationResult && (() => {
                const table = formatComparisonTable(validationResult)
                const imp   = improvementPercent(
                  validationResult.fixed.summary.avgQueue,
                  validationResult.rl.summary.avgQueue,
                  'queue'
                )
                return (
                  <div>
                    <p>RL vs Fixed: queue reduced by <strong>{imp}</strong></p>
                    <table>
                      <thead><tr>{table.headers.map(h => <th key={h}>{h}</th>)}</tr></thead>
                      <tbody>{table.rows.map((row, i) => (
                        <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
                      ))}</tbody>
                    </table>
                  </div>
                )
              })()}
            </div>
          )}
        </section>

        <section className="card sim-card" aria-label="Traffic Simulation">
          <div className="sim-topbar">
            <div><span className="label">Mode</span><strong>{modeLabel}</strong></div>
            <div><span className="label">Signal</span><strong>{phaseLabel}</strong></div>
            <div><span className="label">Runtime</span><strong>{view.simTime.toFixed(1)}s</strong></div>
          </div>

          <div className="sim-canvas-wrap">
            <canvas
              ref={canvasRef} width={WORLD.width} height={WORLD.height}
              className="sim-canvas" aria-label="Traffic intersection simulation"
            />

            <div className="signal-arrow-overlay" aria-hidden="true">
              {signalHeads.map((signal) => {
                const stage = parseSignalPhase(view.phase)
                const active = stage.axis === signal.axis && view.phase_type === 'green'
                const MovementIcon = getApproachIcon(signal.direction, active ? stage.movement : 'straight')
                const bodyX = signal.x + signal.armX
                const bodyY = signal.y + signal.armY
                return (
                  <div
                    key={signal.label}
                    className={`signal-arrow-badge ${active ? 'active' : 'inactive'}`}
                    style={{
                      left: `${((bodyX + WORLD.width  / 2) / WORLD.width)  * 100}%`,
                      top:  `${((bodyY + 82 + WORLD.height / 2) / WORLD.height) * 100}%`,
                    }}
                  >
                    <MovementIcon className="signal-arrow-icon" />
                  </div>
                )
              })}
            </div>
          </div>
        </section>

        <section className="card metrics" aria-label="Performance Metrics">
          <h2>Optimization Metrics</h2>
          <div className="stats-grid">
            <article><span>Vehicles Passed</span><strong>{view.completed}</strong></article>
            <article><span>Live Vehicles</span><strong>{view.liveCars}</strong></article>
            <article><span>Avg Wait</span><strong>{view.avgWait.toFixed(2)}s</strong></article>
            <article><span>Efficiency</span><strong>{view.efficiency.toFixed(1)}/min</strong></article>
            <article><span>NS Queue</span><strong>{view.queues.NS}</strong></article>
            <article><span>EW Queue</span><strong>{view.queues.EW}</strong></article>
          </div>
        </section>
      </main>

      <footer className="footnote">
        Sprite trucks follow lane-aware straight, right, and left-turn paths with solid-body spacing to prevent intersections.
      </footer>
    </div>
  )
}

export default App