function computeRollingMean(values, window) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1)
    return slice.reduce((a, b) => a + b, 0) / slice.length
  })
}

function computeSummaryStats(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mean   = values.reduce((a, b) => a + b, 0) / values.length
  const stddev = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length)
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)]
  return { mean, stddev, min: sorted[0], max: sorted[sorted.length - 1], median }
}

function fmt(mean, std) { return `${mean.toFixed(1)} ± ${std.toFixed(1)}` }

function formatComparisonTable(v) {
  const f = v.fixed.summary, a = v.adaptive.summary, r = v.rl.summary
  return {
    headers: ['Metric', 'Fixed', 'Adaptive', 'RL'],
    rows: [
      ['Avg queue length',  fmt(f.avgQueue, f.stdQueue),       fmt(a.avgQueue, a.stdQueue),       fmt(r.avgQueue, r.stdQueue)],
      ['Total switches',    fmt(f.avgSwitches, f.stdSwitches), fmt(a.avgSwitches, a.stdSwitches), fmt(r.avgSwitches, r.stdSwitches)],
      ['Avg reward/ep',     fmt(f.avgReward, f.stdReward),     fmt(a.avgReward, a.stdReward),     fmt(r.avgReward, r.stdReward)],
      ['Vehicles cleared',  fmt(f.throughput, f.stdThroughput),fmt(a.throughput, a.stdThroughput),fmt(r.throughput, r.stdThroughput)],
    ],
  }
}

function improvementPercent(baseline, rl, metric) {
  if (metric === 'queue' || metric === 'wait')
    return ((baseline - rl) / baseline * 100).toFixed(1) + '%'
  return ((rl - baseline) / Math.abs(baseline) * 100).toFixed(1) + '%'
}

export { computeRollingMean, computeSummaryStats, formatComparisonTable, improvementPercent }