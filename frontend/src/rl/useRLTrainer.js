import { useState, useRef } from 'react'
import { saveToLocalStorage } from './modelStorage.js'

export function useRLTrainer() {
  const workerRef = useRef(null)
  const [trainingState, setTrainingState] = useState('idle')
  const [progress, setProgress] = useState({ ep: 0, totalEpisodes: 200, epsilon: '1.000', episodeReward: '0.00', avgQueue: '0.00', percentComplete: 0 })
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)

  function startTraining(configOverrides = {}) {
    setTrainingState('training')
    setError(null)
    setResult(null)

    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (e) => {
      const msg = e.data
      if (msg.type === 'progress') {
        setProgress({
          ep:              msg.ep,
          totalEpisodes:   msg.totalEpisodes,
          epsilon:         msg.epsilon,
          episodeReward:   msg.episodeReward,
          avgQueue:        msg.avgQueue,
          percentComplete: Math.round((msg.ep / msg.totalEpisodes) * 100),
        })
      } else if (msg.type === 'done') {
        saveToLocalStorage(msg.weights)
        setResult({
          weights: msg.weights,
          episodeMetrics: msg.episodeMetrics,
          stoppedEarly: msg.stoppedEarly,
          finalEpisode: msg.finalEpisode,
          runInfo: msg.runInfo,
        })
        setTrainingState('done')
        worker.terminate()
        workerRef.current = null
      }
    }

    worker.onerror = (e) => {
      setError(e.message || 'Training worker failed')
      setTrainingState('error')
      worker.terminate()
      workerRef.current = null
    }

    worker.postMessage({ type: 'start', config: configOverrides })
  }

  function abortTraining() {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'abort' })
      setTimeout(() => workerRef.current?.terminate(), 500)
      workerRef.current = null
    }
    setTrainingState('idle')
  }

  return { startTraining, abortTraining, trainingState, progress, result, error }
}