import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import { forwardOptimize, reverseOptimize, listResults, listScenarios } from '../lib/api'
import { useStore } from '../store'
import { CHANNEL_COLORS, fmtCurrency, fmt } from '../lib/utils'

interface ChannelBound {
  channel: string
  min_spend: number
  no_upper_limit: boolean
  max_spend: number | null
}

export default function OptimizePage() {
  const sessionId = useStore((s) => s.upload.sessionId)
  const activeModelId = useStore((s) => s.activeModelId)

  const [mode, setMode] = useState<'forward' | 'reverse'>('forward')
  const [budget, setBudget] = useState(500000)
  const [targetConversions, setTargetConversions] = useState(10000)
  const [result, setResult] = useState<any>(null)
  const [channelBounds, setChannelBounds] = useState<ChannelBound[]>([])
  const [error, setError] = useState<string | null>(null)

  const { data: runs = [] } = useQuery({
    queryKey: ['results', sessionId],
    queryFn: () => listResults(sessionId!),
    enabled: !!sessionId,
  })

  const { data: scenarios = [] } = useQuery({
    queryKey: ['scenarios', sessionId],
    queryFn: () => listScenarios(sessionId!),
    enabled: !!sessionId,
  })

  const completeRuns = (runs as any[]).filter((r: any) => r.status === 'complete')
  const modelId = activeModelId ?? completeRuns[0]?.id

  const activeRun = completeRuns.find((r: any) => r.id === modelId)
  const channels: string[] = activeRun?.contributions ? Object.keys(activeRun.contributions) : []

  // Init channel bounds when channels available
  const ensureBounds = () => {
    if (channels.length > 0 && channelBounds.length === 0) {
      setChannelBounds(channels.map((ch) => ({
        channel: ch,
        min_spend: 0,
        no_upper_limit: true,
        max_spend: null,
      })))
    }
  }

  const optimizeMutation = useMutation({
    mutationFn: (payload: any) =>
      mode === 'forward' ? forwardOptimize(payload) : reverseOptimize(payload),
    onSuccess: (data) => setResult(data),
    onError: (e: any) => setError(e.response?.data?.detail ?? 'Optimization failed'),
  })

  const handleRun = () => {
    if (!sessionId || !modelId) return
    setError(null)
    ensureBounds()
    const payload = {
      session_id: sessionId,
      model_run_id: modelId,
      channel_bounds: channelBounds,
      ...(mode === 'forward' ? { total_budget: budget } : { target_conversions: targetConversions }),
    }
    optimizeMutation.mutate(payload)
  }

  const pieData = result
    ? Object.entries(result.channel_allocation).map(([ch, v]) => ({ name: ch, value: Number(v) }))
    : []

  const barData = result
    ? channels.map((ch) => ({
        channel: ch,
        current: activeRun?.contributions?.[ch] ?? 0,
        optimized: result.channel_allocation[ch] ?? 0,
      }))
    : []

  if (!sessionId) {
    return (
      <div className="p-8">
        <div className="card text-center text-gray-500">Please upload data and run a model first.</div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader title="Budget Optimizer" subtitle="Forward (budget → conversions) or reverse (target → min spend)" />

      <div className="p-8 space-y-6">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">{error}</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Config panel */}
          <div className="space-y-5">
            {/* Mode */}
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-3">Optimization mode</h3>
              <div className="space-y-2">
                {([
                  { v: 'forward', label: 'Forward', desc: 'Maximize conversions for a given budget' },
                  { v: 'reverse', label: 'Reverse', desc: 'Minimize spend to hit a conversion target' },
                ] as const).map(({ v, label, desc }) => (
                  <label
                    key={v}
                    className={`block border rounded-lg p-3 cursor-pointer ${mode === v ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}
                  >
                    <input type="radio" name="mode" value={v} checked={mode === v} onChange={() => setMode(v)} className="sr-only" />
                    <div className="font-medium text-sm">{label}</div>
                    <div className="text-xs text-gray-500">{desc}</div>
                  </label>
                ))}
              </div>
            </div>

            {/* Input */}
            <div className="card">
              {mode === 'forward' ? (
                <div>
                  <label className="label">Total budget</label>
                  <input
                    type="number"
                    className="input"
                    value={budget}
                    step={10000}
                    onChange={(e) => setBudget(Number(e.target.value))}
                  />
                </div>
              ) : (
                <div>
                  <label className="label">Target conversions</label>
                  <input
                    type="number"
                    className="input"
                    value={targetConversions}
                    step={100}
                    onChange={(e) => setTargetConversions(Number(e.target.value))}
                  />
                </div>
              )}
            </div>

            {/* Channel bounds */}
            {channels.length > 0 && (
              <div className="card">
                <h3 className="font-semibold text-gray-900 mb-3">Channel bounds</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {channels.map((ch, i) => {
                    const bound = channelBounds.find((b) => b.channel === ch) ?? {
                      channel: ch, min_spend: 0, no_upper_limit: true, max_spend: null,
                    }
                    const updateBound = (field: keyof ChannelBound, val: any) => {
                      setChannelBounds((prev) => {
                        const existing = prev.findIndex((b) => b.channel === ch)
                        const updated = { ...bound, [field]: val }
                        if (existing >= 0) return prev.map((b, idx) => idx === existing ? updated : b)
                        return [...prev, updated]
                      })
                    }
                    return (
                      <div key={ch} className="border rounded-lg p-3 space-y-2">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ background: CHANNEL_COLORS[i % CHANNEL_COLORS.length] }} />
                          <span className="text-sm font-medium">{ch}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-xs text-gray-500">Min spend</label>
                            <input
                              type="number"
                              className="input text-xs"
                              value={bound.min_spend}
                              onChange={(e) => updateBound('min_spend', Number(e.target.value))}
                            />
                          </div>
                          <div>
                            <label className="text-xs text-gray-500">Max spend</label>
                            <div className="flex items-center gap-1">
                              <input
                                type="number"
                                className="input text-xs"
                                value={bound.max_spend ?? ''}
                                disabled={bound.no_upper_limit}
                                onChange={(e) => updateBound('max_spend', Number(e.target.value))}
                              />
                            </div>
                            <label className="flex items-center gap-1 mt-1">
                              <input
                                type="checkbox"
                                checked={bound.no_upper_limit}
                                onChange={(e) => updateBound('no_upper_limit', e.target.checked)}
                              />
                              <span className="text-xs text-gray-500">No limit</span>
                            </label>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            <button
              className="btn-primary w-full"
              onClick={handleRun}
              disabled={optimizeMutation.isPending || !modelId}
            >
              {optimizeMutation.isPending ? <span className="flex items-center justify-center gap-2"><Spinner />Running...</span> : 'Run Optimization'}
            </button>
          </div>

          {/* Results */}
          <div className="lg:col-span-2 space-y-5">
            {result && (
              <>
                {/* Summary metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="card text-center">
                    <div className="text-2xl font-bold text-blue-600">{fmtCurrency(result.total_spend)}</div>
                    <div className="text-xs text-gray-500 mt-1">Total spend</div>
                  </div>
                  <div className="card text-center">
                    <div className="text-2xl font-bold text-green-600">{fmt(result.expected_conversions, 0)}</div>
                    <div className="text-xs text-gray-500 mt-1">Expected conversions</div>
                  </div>
                  <div className="card text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {result.total_spend > 0 ? fmt(result.expected_conversions / result.total_spend * 1000, 2) : '—'}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Conv / £1K</div>
                  </div>
                </div>

                {/* Pie */}
                <div className="card">
                  <h3 className="font-semibold mb-4">Spend allocation</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}>
                        {pieData.map((_, i) => <Cell key={i} fill={CHANNEL_COLORS[i % CHANNEL_COLORS.length]} />)}
                      </Pie>
                      <Tooltip formatter={(v: number) => fmtCurrency(v)} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Comparison bar */}
                <div className="card">
                  <h3 className="font-semibold mb-4">Optimized vs current spend</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={barData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v: number) => fmtCurrency(v)} />
                      <Legend />
                      <Bar dataKey="current" name="Current" fill="#94a3b8" />
                      <Bar dataKey="optimized" name="Optimized" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}

            {/* Saved scenarios */}
            {(scenarios as any[]).length > 0 && (
              <div className="card">
                <h3 className="font-semibold mb-3">Saved scenarios</h3>
                <div className="space-y-2">
                  {(scenarios as any[]).map((s: any) => (
                    <div key={s.id} className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-2 text-sm">
                      <span className="font-medium">{s.name}</span>
                      <span className="text-gray-500 text-xs capitalize">{s.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
