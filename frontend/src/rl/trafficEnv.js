const PHASES = ['NS_straight', 'NS_left', 'EW_straight', 'EW_left']
const MIN_GREEN_STEPS = 10
const MAX_GREEN_STEPS = 60
const EPISODE_LENGTH  = 3600
const FLOW_RATE       = 2
const STATE_QUEUE_NORM = 60

class TrafficEnv {
  constructor(config = {}) {
    this.spawnRate          = config.spawnRate          ?? 0.7
    this.leftTurnMultiplier = config.leftTurnMultiplier ?? 1.0
    this.episodeLength      = config.episodeLength      ?? EPISODE_LENGTH
    this._rng               = null  // set to seeded fn during validation
    this.reset()
  }

  reset() {
    this.queues               = { NS: 0, EW: 0, NSL: 0, EWL: 0 }
    this.currentPhase         = 0
    this.timeInPhase          = 0
    this.totalSteps           = 0
    this.totalWait            = 0
    this.totalSwitches        = 0
    this.totalVehiclesCleared = 0
    return this._getState()
  }

  step(agentAction) {
    const locked       = this.timeInPhase < MIN_GREEN_STEPS
    const forcedSwitch = this.timeInPhase >= MAX_GREEN_STEPS
    let action = agentAction
    if (locked)       action = 0
    if (forcedSwitch) action = 1
    const switched = (action === 1)

    const queuesBefore = this._getQueueArray()

    if (switched) {
      this.currentPhase = (this.currentPhase + 1) % 4
      this.timeInPhase  = 0
      this.totalSwitches++
    } else {
      this.timeInPhase++
    }

    const cleared = this._simTick()

    const queuesAfter = this._getQueueArray()
    const totalBefore = queuesBefore.reduce((a, b) => a + b, 0)
    const totalAfter  = queuesAfter.reduce((a, b) => a + b, 0)

    const queueDelta = totalBefore - totalAfter
    const nsPressure = this.queues.NS + this.queues.NSL
    const ewPressure = this.queues.EW + this.queues.EWL
    const balancePenalty = -0.01 * Math.abs(nsPressure - ewPressure)
    const pressurePenalty = -0.015 * totalAfter
    const switchPenalty = switched ? -0.08 : 0.0
    const throughputBonus = 0.25 * cleared
    const reward = (0.5 * queueDelta) + throughputBonus + pressurePenalty + balancePenalty + switchPenalty

    this.totalSteps++
    this.totalWait += totalAfter
    const done = this.totalSteps >= this.episodeLength

    return {
      nextState: this._getState(),
      reward,
      done,
      info: { switched, locked, forcedSwitch, queuesAfter, totalAfter, currentPhase: this.currentPhase },
    }
  }

  getEpisodeSummary() {
    return {
      avgQueueLength:       this.totalWait / this.totalSteps,
      totalSwitches:        this.totalSwitches,
      totalVehiclesCleared: this.totalVehiclesCleared,
      stepsPerPhase:        this.totalSteps / Math.max(1, this.totalSwitches),
    }
  }

  _getState() {
    const clamp01 = (v) => Math.max(0, Math.min(1, v))
    return [
      clamp01(this.queues.NS / STATE_QUEUE_NORM),
      clamp01(this.queues.EW / STATE_QUEUE_NORM),
      clamp01(this.queues.NSL / STATE_QUEUE_NORM),
      clamp01(this.queues.EWL / STATE_QUEUE_NORM),
      this.currentPhase / Math.max(1, PHASES.length - 1),
      Math.min(this.timeInPhase / MAX_GREEN_STEPS, 1.0),
    ]
  }

  _getQueueArray() {
    return [this.queues.NS, this.queues.EW, this.queues.NSL, this.queues.EWL]
  }

  _rand() {
    return this._rng ? this._rng() : Math.random()
  }

  _simTick() {
    const leftRate = this.spawnRate * 0.3 * this.leftTurnMultiplier
    this.queues.NS  += this._rand() < this.spawnRate ? 1 : 0
    this.queues.EW  += this._rand() < this.spawnRate ? 1 : 0
    this.queues.NSL += this._rand() < leftRate       ? 1 : 0
    this.queues.EWL += this._rand() < leftRate       ? 1 : 0

    const phase = PHASES[this.currentPhase]
    let cleared = 0
    if (phase === 'NS_straight') {
      const b = this.queues.NS; this.queues.NS = Math.max(0, this.queues.NS - FLOW_RATE); cleared = b - this.queues.NS
    } else if (phase === 'NS_left') {
      const b = this.queues.NSL; this.queues.NSL = Math.max(0, this.queues.NSL - FLOW_RATE); cleared = b - this.queues.NSL
    } else if (phase === 'EW_straight') {
      const b = this.queues.EW; this.queues.EW = Math.max(0, this.queues.EW - FLOW_RATE); cleared = b - this.queues.EW
    } else if (phase === 'EW_left') {
      const b = this.queues.EWL; this.queues.EWL = Math.max(0, this.queues.EWL - FLOW_RATE); cleared = b - this.queues.EWL
    }
    this.totalVehiclesCleared += cleared
    return cleared
  }
}
export { TrafficEnv, PHASES, MIN_GREEN_STEPS, MAX_GREEN_STEPS, EPISODE_LENGTH, STATE_QUEUE_NORM }