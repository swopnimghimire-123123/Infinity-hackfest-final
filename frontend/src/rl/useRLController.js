import { useState, useRef } from 'react'
import { RLInference } from './inference.js'
import { loadFromLocalStorage, getModelMeta } from './modelStorage.js'

export function useRLController() {
  const inferenceRef = useRef(null)
  const [isReady,   setIsReady]   = useState(false)
  const [modelMeta, setModelMeta] = useState(null)
  const [loadedWeights, setLoadedWeights] = useState(null)

  function loadModel(weightsObj) {
    inferenceRef.current = new RLInference(weightsObj)
    setIsReady(inferenceRef.current.isLoaded())
    setLoadedWeights(weightsObj)
    setModelMeta(getModelMeta() ?? {
      savedAt: weightsObj.trainedAt ?? Date.now(),
      totalSteps: weightsObj.totalSteps ?? 0,
      version: weightsObj.version ?? 1,
    })
  }

  function loadFromStorage() {
    const obj = loadFromLocalStorage()
    if (!obj) return false
    loadModel(obj)
    return true
  }

  function getNextAction(state) {
    if (!isReady || !inferenceRef.current) return null
    return inferenceRef.current.selectAction(state)
  }

  return { loadModel, loadFromStorage, getNextAction, isReady, modelMeta, loadedWeights }
}