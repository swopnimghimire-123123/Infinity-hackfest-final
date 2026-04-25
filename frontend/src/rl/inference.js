class RLInference {
  constructor(weightsObj) {
    this._weights = weightsObj
    this._valid   = this._check()
  }

  _check() {
    try {
      const r = this._run([0, 0, 0, 0, 0, 0])
      return isFinite(r.qHold) && isFinite(r.qSwitch)
    } catch { return false }
  }

  _run(state) {
    const w     = this._weights
    const sizes = [w.stateSize, ...w.hiddenSizes, w.actionSize]
    let x = Array.from(state)
    for (let l = 0; l < w.layers.length; l++) {
      const { W, b } = w.layers[l]
      const rows = sizes[l], cols = sizes[l + 1]
      const out = new Array(cols)
      for (let j = 0; j < cols; j++) {
        let sum = b[j]
        for (let i = 0; i < rows; i++) sum += x[i] * W[i * cols + j]
        out[j] = l < w.layers.length - 1 ? Math.max(0, sum) : sum
      }
      x = out
    }
    return { qHold: x[0], qSwitch: x[1] }
  }

  selectAction(state) {
    const { qHold, qSwitch } = this._run(state)
    return { action: qSwitch > qHold ? 1 : 0, qHold, qSwitch }
  }

  getQValues(state) {
    return this._run(state)
  }

  isLoaded() { return this._valid }
}

export { RLInference }