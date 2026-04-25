import { TrafficEnv } from './trafficEnv.js'
import { RLInference } from './inference.js'

// Seeded PRNG — mulberry32
function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

const SEEDS = [42, 137, 256, 891, 1024, 2048, 3141, 5000, 7777, 9999]
const EPISODE_LENGTH = 3600

function runController(controller, rlInference, seed) {
  const rng = mulberry32(seed)
  const env = new TrafficEnv({ spawnRate: 0.7 })
  env._rng  = rng

  let state = env.reset()
  let episodeReward = 0
  let maxQueueSeen  = 0
  let done = false

  while (!done) {
    let action = 0

    if (controller === 'fixed') {
      action = env.timeInPhase >= 30 ? 1 : 0
    } else if (controller === 'adaptive') {
      const q = env.queues
      const activeQueue = [q.NS, q.NSL, q.EW, q.EWL][env.currentPhase]
      action = (activeQueue < 2 || env.timeInPhase >= 40) ? 1 : 0
    } else if (controller === 'rl') {
      const { action: a } = rlInference.selectAction(state)
      action = a
    }

    const { nextState, reward, done: d } = env.step(action)
    episodeReward += reward
    const total = env.queues.NS + env.queues.EW + env.queues.NSL + env.queues.EWL
    if (total > maxQueueSeen) maxQueueSeen = total
    state = nextState
    done  = d
  }

  const summary = env.getEpisodeSummary()
  return {
    seed, controller,
    avgQueue:             summary.avgQueueLength,
    totalSwitches:        summary.totalSwitches,
    episodeReward,
    totalVehiclesCleared: summary.totalVehiclesCleared,
    maxQueueSeen,
  }
}

async function runValidation(rlWeightsObj) {
  const rl = new RLInference(rlWeightsObj)
  const results = { fixed: [], adaptive: [], rl: [] }

  for (const seed of SEEDS) {
    results.fixed.push(runController('fixed', null, seed))
    results.adaptive.push(runController('adaptive', null, seed))
    results.rl.push(runController('rl', rl, seed))
  }

  function summarize(episodes) {
    const avg  = (key) => episodes.reduce((a, e) => a + e[key], 0) / episodes.length
    const std  = (key) => {
      const m = avg(key)
      return Math.sqrt(episodes.reduce((a, e) => a + (e[key] - m) ** 2, 0) / episodes.length)
    }
    return {
      avgQueue:    avg('avgQueue'),
      avgSwitches: avg('totalSwitches'),
      avgReward:   avg('episodeReward'),
      throughput:  avg('totalVehiclesCleared'),
      stdQueue:    std('avgQueue'),
      stdSwitches: std('totalSwitches'),
      stdReward:   std('episodeReward'),
      stdThroughput: std('totalVehiclesCleared'),
    }
  }

  return {
    fixed:    { episodes: results.fixed,    summary: summarize(results.fixed) },
    adaptive: { episodes: results.adaptive, summary: summarize(results.adaptive) },
    rl:       { episodes: results.rl,       summary: summarize(results.rl) },
  }
}

export { runValidation }