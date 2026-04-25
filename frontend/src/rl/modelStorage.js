// Python reconstruction (for backend use):
//
// import json, numpy as np
// def load_model(path):
//     with open(path) as f: m = json.load(f)
//     layers = []
//     shapes = [(6,32),(32,32),(32,2)]
//     for i, layer in enumerate(m['layers']):
//         rows, cols = shapes[i]
//         W = np.array(layer['W']).reshape(rows, cols)
//         b = np.array(layer['b'])
//         layers.append((W, b))
//     return layers
//
// def forward(layers, state):
//     x = np.array(state, dtype=np.float32)
//     for i, (W, b) in enumerate(layers):
//         x = x @ W + b
//         if i < len(layers) - 1:
//             x = np.maximum(0, x)
//     return int(np.argmax(x))

const MODEL_KEY = 'rl_traffic_model_v1'

function saveToLocalStorage(weightsObj) {
  try {
    localStorage.setItem(MODEL_KEY, JSON.stringify(weightsObj))
    localStorage.setItem(MODEL_KEY + '_meta', JSON.stringify({
      savedAt:    Date.now(),
      totalSteps: weightsObj.totalSteps,
      version:    weightsObj.version,
    }))
    return true
  } catch (e) {
    if (e.name === 'QuotaExceededError') return false
    throw e
  }
}

function loadFromLocalStorage() {
  const raw = localStorage.getItem(MODEL_KEY)
  return raw ? JSON.parse(raw) : null
}

function exportToFile(weightsObj, filename = 'rl_traffic_model.json') {
  const blob = new Blob([JSON.stringify(weightsObj, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

async function importFromFile(file) {
  const text = await file.text()
  const obj  = JSON.parse(text)
  if (obj.version !== 1)    throw new Error(`Invalid model version: ${obj.version}`)
  if (obj.stateSize !== 6)  throw new Error(`Expected stateSize 6, got ${obj.stateSize}`)
  return obj
}

function getModelMeta() {
  const raw = localStorage.getItem(MODEL_KEY + '_meta')
  return raw ? JSON.parse(raw) : null
}

function clearModel() {
  localStorage.removeItem(MODEL_KEY)
  localStorage.removeItem(MODEL_KEY + '_meta')
}

export { saveToLocalStorage, loadFromLocalStorage, exportToFile, importFromFile, getModelMeta, clearModel }