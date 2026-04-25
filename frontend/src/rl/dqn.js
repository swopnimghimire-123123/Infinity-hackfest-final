function heRandom(fanIn, rand = Math.random) {
  // Box-Muller for normal distribution
  const u1 = Math.max(1e-8, rand()), u2 = rand()
  const std = Math.sqrt(2 / fanIn)
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std
}

class DQN {
  constructor({ stateSize = 6, actionSize = 2, hiddenSizes = [32, 32],
                lr = 0.001, gamma = 0.97, batchSize = 64,
                bufferSize = 10000, targetUpdateFreq = 200,
                gradClip = 1.0, huberDelta = 1.0, rewardClip = 5.0,
                random = Math.random } = {}) {
    this.stateSize       = stateSize
    this.actionSize      = actionSize
    this.hiddenSizes     = hiddenSizes
    this.lr              = lr
    this.gamma           = gamma
    this.batchSize       = batchSize
    this.bufferSize      = bufferSize
    this.targetUpdateFreq = targetUpdateFreq
    this.gradClip        = gradClip
    this.huberDelta      = huberDelta
    this.rewardClip      = rewardClip
    this._rand           = random
    this.stepCount       = 0

    // Replay buffer (circular)
    this.buffer      = []
    this.bufferIndex = 0

    // Networks
    this.online = this._buildNet(stateSize, hiddenSizes, actionSize)
    this.target = this._buildNet(stateSize, hiddenSizes, actionSize)
    this._syncTarget()
  }

  _buildNet(inputSize, hiddenSizes, outputSize) {
    const sizes = [inputSize, ...hiddenSizes, outputSize]
    const layers = []
    for (let i = 0; i < sizes.length - 1; i++) {
      const rows = sizes[i], cols = sizes[i + 1]
      const W  = new Float32Array(rows * cols).map(() => heRandom(rows, this._rand))
      const b  = new Float32Array(cols)
      const mW = new Float32Array(rows * cols)
      const vW = new Float32Array(rows * cols)
      const mb = new Float32Array(cols)
      const vb = new Float32Array(cols)
      layers.push({ W, b, mW, vW, mb, vb, t: 0, rows, cols })
    }
    return { layers }
  }

  _forward(net, state) {
    let x = new Float32Array(state)
    for (let l = 0; l < net.layers.length; l++) {
      const { W, b, rows, cols } = net.layers[l]
      const out = new Float32Array(cols)
      for (let j = 0; j < cols; j++) {
        let sum = b[j]
        for (let i = 0; i < rows; i++) sum += x[i] * W[i * cols + j]
        out[j] = l < net.layers.length - 1 ? Math.max(0, sum) : sum  // ReLU or linear
      }
      x = out
    }
    return x
  }

  selectAction(state, epsilon) {
    if (this._rand() < epsilon) return this._rand() < 0.5 ? 0 : 1
    const q = this._forward(this.online, state)
    return q[0] >= q[1] ? 0 : 1
  }

  storeTransition(s, a, r, sNext, done) {
    const entry = { s: Float32Array.from(s), a, r, sNext: Float32Array.from(sNext), done }
    if (this.buffer.length < this.bufferSize) {
      this.buffer.push(entry)
    } else {
      this.buffer[this.bufferIndex] = entry
    }
    this.bufferIndex = (this.bufferIndex + 1) % this.bufferSize
  }

  train() {
    if (this.buffer.length < this.batchSize) return

    const batch = this._sampleBatch()
    for (const { s, a, r, sNext, done } of batch) {
      const clippedReward = Math.max(-this.rewardClip, Math.min(this.rewardClip, r))
      const onlineNext = this._forward(this.online, sNext)
      const bestNextAction = onlineNext[0] >= onlineNext[1] ? 0 : 1
      const targetQ = this._forward(this.target, sNext)
      const bootstrap = targetQ[bestNextAction]
      const target = done ? clippedReward : clippedReward + this.gamma * bootstrap
      this._gradStep(s, a, target)
    }

    this.stepCount++
    if (this.stepCount % this.targetUpdateFreq === 0) this._syncTarget()
  }

  _gradStep(state, action, target) {
    // Forward pass — store activations for backprop
    const activations = [new Float32Array(state)]
    let x = new Float32Array(state)
    for (let l = 0; l < this.online.layers.length; l++) {
      const { W, b, rows, cols } = this.online.layers[l]
      const out = new Float32Array(cols)
      for (let j = 0; j < cols; j++) {
        let sum = b[j]
        for (let i = 0; i < rows; i++) sum += x[i] * W[i * cols + j]
        out[j] = l < this.online.layers.length - 1 ? Math.max(0, sum) : sum
      }
      activations.push(out)
      x = out
    }

    const qVals = activations[activations.length - 1]
    const error = qVals[action] - target
    const absErr = Math.abs(error)
    const huberGrad = absErr <= this.huberDelta
      ? error
      : this.huberDelta * Math.sign(error)

    // Guard against NaN
    if (!isFinite(error)) return

    // Backprop — delta starts at output layer
    let delta = new Float32Array(this.actionSize)
    delta[action] = huberGrad

    for (let l = this.online.layers.length - 1; l >= 0; l--) {
      const layer  = this.online.layers[l]
      const input  = activations[l]
      const output = activations[l + 1]
      const { rows, cols } = layer

      const dInput = new Float32Array(rows)

      layer.t++
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8
      const bc1 = 1 - Math.pow(beta1, layer.t)
      const bc2 = 1 - Math.pow(beta2, layer.t)

      for (let j = 0; j < cols; j++) {
        // Apply ReLU gate on hidden layers
        const gate = l < this.online.layers.length - 1 ? (output[j] > 0 ? 1 : 0) : 1
        const gjRaw = delta[j] * gate
        const gj = Math.max(-this.gradClip, Math.min(this.gradClip, gjRaw))

        // Bias update (Adam)
        layer.mb[j] = beta1 * layer.mb[j] + (1 - beta1) * gj
        layer.vb[j] = beta2 * layer.vb[j] + (1 - beta2) * gj * gj
        const mHatB = layer.mb[j] / bc1
        const vHatB = layer.vb[j] / bc2
        layer.b[j] -= this.lr * mHatB / (Math.sqrt(vHatB) + eps)

        for (let i = 0; i < rows; i++) {
          const gradRaw = gj * input[i]
          const grad = Math.max(-this.gradClip, Math.min(this.gradClip, gradRaw))
          const idx  = i * cols + j
          const wOld = layer.W[idx]

          // Weight update (Adam)
          layer.mW[idx] = beta1 * layer.mW[idx] + (1 - beta1) * grad
          layer.vW[idx] = beta2 * layer.vW[idx] + (1 - beta2) * grad * grad
          const mHat = layer.mW[idx] / bc1
          const vHat = layer.vW[idx] / bc2
          const upd  = this.lr * mHat / (Math.sqrt(vHat) + eps)
          if (isFinite(upd)) layer.W[idx] -= upd

          dInput[i] += gj * wOld
        }
      }
      delta = dInput
    }
  }

  _syncTarget() {
    for (let l = 0; l < this.online.layers.length; l++) {
      this.target.layers[l].W.set(this.online.layers[l].W)
      this.target.layers[l].b.set(this.online.layers[l].b)
    }
  }

  _sampleBatch() {
    const out = []
    const len = this.buffer.length
    for (let i = 0; i < this.batchSize; i++) {
      out.push(this.buffer[Math.floor(this._rand() * len)])
    }
    return out
  }

  exportWeights() {
    return {
      version:     1,
      stateSize:   this.stateSize,
      actionSize:  this.actionSize,
      hiddenSizes: this.hiddenSizes,
      trainedAt:   Date.now(),
      totalSteps:  this.stepCount,
      layers: this.online.layers.map(l => ({
        W: Array.from(l.W),
        b: Array.from(l.b),
      })),
    }
  }

  static importWeights(obj) {
    const agent = new DQN({
      stateSize:  obj.stateSize,
      actionSize: obj.actionSize,
      hiddenSizes: obj.hiddenSizes,
    })
    for (let l = 0; l < obj.layers.length; l++) {
      agent.online.layers[l].W = new Float32Array(obj.layers[l].W)
      agent.online.layers[l].b = new Float32Array(obj.layers[l].b)
    }
    agent._syncTarget()
    agent.stepCount = obj.totalSteps ?? 0
    return agent
  }

  static forwardOnly(weightsObj, state) {
    const sizes = [weightsObj.stateSize, ...weightsObj.hiddenSizes, weightsObj.actionSize]
    let x = new Float32Array(state)
    for (let l = 0; l < weightsObj.layers.length; l++) {
      const { W, b } = weightsObj.layers[l]
      const rows = sizes[l], cols = sizes[l + 1]
      const out = new Float32Array(cols)
      for (let j = 0; j < cols; j++) {
        let sum = b[j]
        for (let i = 0; i < rows; i++) sum += x[i] * W[i * cols + j]
        out[j] = l < weightsObj.layers.length - 1 ? Math.max(0, sum) : sum
      }
      x = out
    }
    return x[0] >= x[1] ? 0 : 1
  }
}
export { DQN }