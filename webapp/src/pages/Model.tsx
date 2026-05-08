import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Plus, Trash2 } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import StatusBadge from '../components/StatusBadge'
import { fitModel, getModelStatus, getUploadColumns } from '../lib/api'
import { useStore } from '../store'

interface HaloPair {
  type: 'channel' | 'campaign'
  a: string
  b: string
  subtract_campaign_spend: boolean
}

function useJobPoller(jobId: number | null, onComplete: (result: any) => void) {
  useEffect(() => {
    if (!jobId) return
    const timer = setInterval(async () => {
      const status = await getModelStatus(jobId)
      if (status.status === 'complete' || status.status === 'failed') {
        clearInterval(timer)
        onComplete(status)
      }
    }, 3000)
    return () => clearInterval(timer)
  }, [jobId, onComplete])
}

export default function ModelPage() {
  const navigate = useNavigate()
  const sessionId = useStore((s) => s.upload.sessionId)
  const hasCampaignData = useStore((s) => s.upload.hasCampaignData)
  const runningJobId = useStore((s) => s.runningJobId)
  const setRunningJobId = useStore((s) => s.setRunningJobId)
  const setActiveModelId = useStore((s) => s.setActiveModelId)

  const [method, setMethod] = useState('map')
  const [samples, setSamples] = useState(1000)
  const [tune, setTune] = useState(1000)
  const [chains, setChains] = useState(2)
  const [targetAccept, setTargetAccept] = useState(0.9)
  const [minHaloSpend, setMinHaloSpend] = useState(100000)
  const [haloPairs, setHaloPairs] = useState<HaloPair[]>([])
  const [haloType, setHaloType] = useState<'channel' | 'campaign'>('channel')
  const [jobStatus, setJobStatus] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const { data: colData } = useQuery({
    queryKey: ['columns', sessionId],
    queryFn: () => getUploadColumns(sessionId!),
    enabled: !!sessionId,
  })

  const channels: string[] = colData?.channels ?? []
  const campaigns: string[] = colData?.campaigns ?? []

  useJobPoller(runningJobId, (result) => {
    setJobStatus(result)
    if (result.status === 'complete') {
      setActiveModelId(result.job_id)
      setRunningJobId(null)
    }
  })

  const fitMutation = useMutation({
    mutationFn: (payload: any) => fitModel(payload),
    onSuccess: (data) => {
      setRunningJobId(data.job_id)
      setJobStatus({ status: 'running', job_id: data.job_id })
    },
    onError: (e: any) => {
      setError(e.response?.data?.detail ?? 'Failed to start model fit')
    },
  })

  const addHaloPair = () => {
    const opts = haloType === 'channel' ? channels : campaigns
    if (opts.length < 2) return
    setHaloPairs((p) => [...p, { type: haloType, a: opts[0], b: opts[1], subtract_campaign_spend: true }])
  }

  const removeHaloPair = (i: number) => setHaloPairs((p) => p.filter((_, idx) => idx !== i))

  const handleFit = () => {
    if (!sessionId) return
    setError(null)
    fitMutation.mutate({
      session_id: sessionId,
      inference_method: method,
      samples,
      tune,
      chains,
      target_accept: targetAccept,
      halo_pairs: haloPairs,
      min_halo_spend: minHaloSpend,
    })
  }

  const isRunning = runningJobId !== null || fitMutation.isPending

  return (
    <div>
      <PageHeader
        title="Configure Model"
        subtitle="Set inference parameters and halo effects"
        actions={
          jobStatus?.status === 'complete' ? (
            <button className="btn-primary" onClick={() => navigate('/results')}>
              View Results
            </button>
          ) : null
        }
      />

      <div className="p-8 space-y-6 max-w-3xl">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">{error}</div>
        )}

        {/* Job status */}
        {jobStatus && (
          <div className="card flex items-center gap-4">
            <StatusBadge status={jobStatus.status} />
            <span className="text-sm text-gray-600">
              {jobStatus.status === 'running' && 'Model is fitting — this may take several minutes...'}
              {jobStatus.status === 'complete' && `Model #${jobStatus.model_num} complete!`}
              {jobStatus.status === 'failed' && 'Model fit failed. Check error details below.'}
            </span>
            {jobStatus.status === 'running' && <Spinner />}
          </div>
        )}

        {/* Inference method */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Inference method</h3>
          <div className="flex gap-4">
            {[
              { v: 'map', label: 'MAP', desc: 'Fast point estimate (~30s)' },
              { v: 'advi', label: 'ADVI', desc: 'Variational inference (~2min)' },
              { v: 'mcmc', label: 'MCMC', desc: 'Full posterior (~10min)' },
            ].map(({ v, label, desc }) => (
              <label
                key={v}
                className={`flex-1 border rounded-lg p-3 cursor-pointer transition-colors ${
                  method === v ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input type="radio" name="method" value={v} checked={method === v} onChange={() => setMethod(v)} className="sr-only" />
                <div className="font-medium text-sm">{label}</div>
                <div className="text-xs text-gray-500 mt-0.5">{desc}</div>
              </label>
            ))}
          </div>
        </div>

        {/* Parameters */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Sampling parameters</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Samples', value: samples, set: setSamples, min: 100 },
              { label: 'Tune', value: tune, set: setTune, min: 100 },
              { label: 'Chains', value: chains, set: setChains, min: 1, max: 8 },
            ].map(({ label, value, set, min, max }) => (
              <div key={label}>
                <label className="label">{label}</label>
                <input
                  type="number"
                  className="input"
                  value={value}
                  min={min}
                  max={max}
                  onChange={(e) => set(Number(e.target.value))}
                />
              </div>
            ))}
            <div>
              <label className="label">Target accept</label>
              <input
                type="number"
                className="input"
                value={targetAccept}
                min={0.5}
                max={0.99}
                step={0.01}
                onChange={(e) => setTargetAccept(Number(e.target.value))}
              />
            </div>
          </div>
        </div>

        {/* Halo effects */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Halo effects</h3>

          <div className="flex items-center gap-4 mb-4">
            <div className="flex gap-2">
              {(['channel', 'campaign'] as const).map((t) => (
                <button
                  key={t}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    haloType === t ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                  onClick={() => setHaloType(t)}
                  disabled={t === 'campaign' && !hasCampaignData}
                >
                  {t === 'channel' ? 'Channel-level' : 'Campaign-level'}
                </button>
              ))}
            </div>
            <button className="btn-secondary flex items-center gap-1.5" onClick={addHaloPair}>
              <Plus size={14} />
              Add pair
            </button>
          </div>

          <div>
            <label className="label">Min spend threshold (exclude low-spend items)</label>
            <input
              type="number"
              className="input w-48"
              value={minHaloSpend}
              step={10000}
              onChange={(e) => setMinHaloSpend(Number(e.target.value))}
            />
          </div>

          {haloPairs.length > 0 && (
            <div className="mt-4 space-y-3">
              {haloPairs.map((pair, i) => {
                const opts = pair.type === 'channel' ? channels : campaigns
                return (
                  <div key={i} className="flex items-center gap-3 bg-gray-50 rounded-lg p-3">
                    <select
                      className="input flex-1"
                      value={pair.a}
                      onChange={(e) => setHaloPairs((p) => p.map((x, idx) => idx === i ? { ...x, a: e.target.value } : x))}
                    >
                      {opts.map((o) => <option key={o}>{o}</option>)}
                    </select>
                    <span className="text-gray-400 text-sm">↔</span>
                    <select
                      className="input flex-1"
                      value={pair.b}
                      onChange={(e) => setHaloPairs((p) => p.map((x, idx) => idx === i ? { ...x, b: e.target.value } : x))}
                    >
                      {opts.map((o) => <option key={o}>{o}</option>)}
                    </select>
                    {pair.type === 'campaign' && (
                      <label className="flex items-center gap-1.5 text-xs text-gray-600 whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={pair.subtract_campaign_spend}
                          onChange={(e) => setHaloPairs((p) => p.map((x, idx) => idx === i ? { ...x, subtract_campaign_spend: e.target.checked } : x))}
                        />
                        Subtract from channel
                      </label>
                    )}
                    <button className="text-red-400 hover:text-red-600" onClick={() => removeHaloPair(i)}>
                      <Trash2 size={16} />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <button
          className="btn-primary w-full py-3 text-base"
          onClick={handleFit}
          disabled={isRunning || !sessionId}
        >
          {isRunning ? (
            <span className="flex items-center justify-center gap-2">
              <Spinner />
              Fitting model...
            </span>
          ) : (
            'Run Model'
          )}
        </button>
      </div>
    </div>
  )
}
