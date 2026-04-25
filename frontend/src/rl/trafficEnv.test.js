import test from 'node:test'
import assert from 'node:assert/strict'
import { TrafficEnv, MIN_GREEN_STEPS, MAX_GREEN_STEPS } from './trafficEnv.js'

test('min green lock prevents early switch', () => {
  const env = new TrafficEnv({ spawnRate: 0, leftTurnMultiplier: 0, episodeLength: 20 })
  env.reset()

  // First few steps should hold regardless of requested switch.
  for (let i = 0; i < MIN_GREEN_STEPS - 1; i++) {
    env.step(1)
    assert.equal(env.currentPhase, 0)
  }
})

test('max green forces a switch even if agent holds', () => {
  const env = new TrafficEnv({ spawnRate: 0, leftTurnMultiplier: 0, episodeLength: 100 })
  env.reset()

  // Move to max green.
  for (let i = 0; i <= MAX_GREEN_STEPS; i++) env.step(0)

  assert.equal(env.currentPhase, 1)
})

test('switch action includes penalty in reward', () => {
  const holdEnv = new TrafficEnv({ spawnRate: 0, leftTurnMultiplier: 0, episodeLength: 30 })
  const switchEnv = new TrafficEnv({ spawnRate: 0, leftTurnMultiplier: 0, episodeLength: 30 })
  holdEnv.reset()
  switchEnv.reset()

  // Advance enough steps to avoid min-green lock.
  for (let i = 0; i < MIN_GREEN_STEPS; i++) {
    holdEnv.step(0)
    switchEnv.step(0)
  }

  const holdReward = holdEnv.step(0).reward
  const switchReward = switchEnv.step(1).reward
  assert.ok(switchReward < holdReward)
})
