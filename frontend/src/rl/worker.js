import { DQN } from './dqn.js'
import { TrafficEnv } from './trafficEnv.js'

const CONFIG = {
  episodes:              200,
  episodeLength:         3600,
  epsilonStart:          1.0,
  epsilonEnd:            0.02,
  epsilonDecayEpisodes:  220,
  lr:                    0.0004,
  gamma:                 0.99,
  batchSize:             128,
  bufferSize:            50000,
  targetUpdateFreq:      800,
  trainingStartSteps:    3000,
  trainEvery:            2,
  hiddenSizes:           [64, 64],
  gradClip:              1.0,
  huberDelta:            1.0,
  rewardClip:            5.0,
  progressInterval:      10,
  convergenceWindow:     20,
  convergenceThreshold:  0.005,
  enableEarlyStopping:   true,
  minEpisodesBeforeEarlyStop: 40,
  spawnRateMin:          0.55,
  spawnRateMax:          0.98,
  leftTurnMultMin:       0.7,
  leftTurnMultMax:       1.8,
  seed:                  12345,
  presetName:            'balanced',
}

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function randUniform(rng, min, max) { return min + rng() * (max - min) }
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length }

function runTraining(cfg) {
  const env   = new TrafficEnv({ spawnRate: cfg.spawnRateMin, episodeLength: cfg.episodeLength })
  const agentRng = mulberry32(((cfg.seed >>> 0) ^ 0xA5A5A5A5) >>> 0)
  const agent = new DQN({
    stateSize:        6,
    actionSize:       2,
    hiddenSizes:      cfg.hiddenSizes,
    lr:               cfg.lr,
    gamma:            cfg.gamma,
    batchSize:        cfg.batchSize,
    bufferSize:       cfg.bufferSize,
    targetUpdateFreq: cfg.targetUpdateFreq,
    gradClip:         cfg.gradClip,
    huberDelta:       cfg.huberDelta,
    rewardClip:       cfg.rewardClip,
    random:           agentRng,
  })

  const allMetrics  = []
  const allRewards  = []
  const runLog      = []
  let totalStepsEver = 0
  let aborted        = false
  let stoppedEarly   = false
  const startedAt    = Date.now()

  self.onmessage = (e) => { if (e.data.type === 'abort') aborted = true }

  for (let ep = 0; ep < cfg.episodes; ep++) {
    if (aborted) break

    const episodeSeed = ((cfg.seed >>> 0) + (ep * 10007)) >>> 0
    const episodeRng = mulberry32(episodeSeed)
    env._rng = episodeRng

    env.spawnRate          = randUniform(episodeRng, cfg.spawnRateMin, cfg.spawnRateMax)
    env.leftTurnMultiplier = randUniform(episodeRng, cfg.leftTurnMultMin, cfg.leftTurnMultMax)

    const progress = Math.min(ep / cfg.epsilonDecayEpisodes, 1)
    const epsilon  = Math.max(cfg.epsilonEnd, cfg.epsilonStart - progress * (cfg.epsilonStart - cfg.epsilonEnd))

    let state = env.reset()
    let episodeReward = 0
    let done = false

    while (!done) {
      const action = agent.selectAction(state, epsilon)
      const { nextState, reward, done: d, info } = env.step(action)

      agent.storeTransition(state, action, reward, nextState, d)
      if (totalStepsEver >= cfg.trainingStartSteps && totalStepsEver % cfg.trainEvery === 0) {
        agent.train()
      }

      state = nextState
      episodeReward += reward
      totalStepsEver++
      done = d
    }

    const summary = env.getEpisodeSummary()
    allRewards.push(episodeReward)
    allMetrics.push({
      ep, epsilon, episodeReward,
      avgQueue:     summary.avgQueueLength,
      switches:     summary.totalSwitches,
      stepsPerPhase: summary.stepsPerPhase,
      spawnRate:    env.spawnRate,
      leftTurnMult: env.leftTurnMultiplier,
      seed:         episodeSeed,
    })

    runLog.push({
      ep,
      reward: Number(episodeReward.toFixed(3)),
      avgQueue: Number(summary.avgQueueLength.toFixed(3)),
      switches: summary.totalSwitches,
      epsilon: Number(epsilon.toFixed(4)),
      seed: episodeSeed,
    })

    // Early stopping is configurable by preset. Thorough disables it to run full budget.
    const canEarlyStop =
      cfg.enableEarlyStopping !== false &&
      ep >= Math.max(cfg.convergenceWindow + 10, cfg.minEpisodesBeforeEarlyStop ?? 0)

    if (canEarlyStop) {
      const recent = allRewards.slice(ep - cfg.convergenceWindow)
      const prev   = allRewards.slice(ep - cfg.convergenceWindow - 10, ep - 10)
      const recentMean = mean(recent)
      const prevMean   = mean(prev)
      const improvement = Math.abs(recentMean - prevMean) / (Math.abs(prevMean) + 1e-8)
      if (improvement < cfg.convergenceThreshold) {
        stoppedEarly = true
        self.postMessage({ type: 'progress', ep, totalEpisodes: cfg.episodes,
          epsilon: epsilon.toFixed(3), episodeReward: episodeReward.toFixed(2),
          avgQueue: summary.avgQueueLength.toFixed(2), switches: summary.totalSwitches })
        break
      }
    }

    if (ep % cfg.progressInterval === 0) {
      self.postMessage({ type: 'progress', ep, totalEpisodes: cfg.episodes,
        epsilon: epsilon.toFixed(3), episodeReward: episodeReward.toFixed(2),
        avgQueue: summary.avgQueueLength.toFixed(2), switches: summary.totalSwitches })
    }
  }

  self.postMessage({
    type:           'done',
    weights:        agent.exportWeights(),
    episodeMetrics: allMetrics,
    stoppedEarly,
    finalEpisode:   allMetrics.length,
    runInfo: {
      startedAt,
      finishedAt: Date.now(),
      durationMs: Date.now() - startedAt,
      config: cfg,
      totalTrainSteps: totalStepsEver,
      log: runLog,
    },
  })
}

self.onmessage = (e) => {
  if (e.data.type === 'start') {
    const cfg = Object.assign({}, CONFIG, e.data.config ?? {})
    runTraining(cfg)
  }
}