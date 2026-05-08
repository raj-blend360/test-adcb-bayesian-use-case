import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Plus, Trash2, ChevronDown } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import StatusBadge from '../components/StatusBadge'
import { runTunedModel, getTuneHistory, listResults } from '../lib/api'
import { useStore } from '../store'

interface Holiday {
  label: string
  start_date: string
  end_date: string
}

export default function TunePage() {
  const navigate = useNavigate()
  const sessionId = useStore((s) => s.upload.sessionId)
  const activeModelId = useStore((s) => s.activeModelId)
  const setRunningJobId = useStore((s) => s.setRunningJobId)

  const [holidays, setHolidays] = useState<Holiday[]>([])
  const [seasonality, setSeasonality] = useState({ quarterly: false, half_yearly: true, annual: true })
  const [jobStatus, setJobStatus] = useState<any>(null)
  const [historyOpen, setHistoryOpen] = useState(false)

  const { data: runsData = [] } = useQuery({
    queryKey: ['results', sessionId],
    queryFn: () => listResults(sessionId!),
    enabled: !!sessionId,
  })

  const { data: history = [] } = useQuery({
    queryKey: ['tune-history', sessionId],
    queryFn: () => getTuneHistory(sessionId!),
    enabled: !!sessionId,
  })

  const baseRun = (runsData as any[]).find((r: any) => r.id === activeModelId)

  const runMutation = useMutation({
    mutationFn: (payload: any) => runTunedModel(payload),
    onSuccess: (data) => {
      setJobStatus({ status: 'running', job_id: data.job_id })
      setRunningJobId(data.job_id)
      // Poll for completion
      const timer = setInterval(async () => {
        const { getModelStatus } = await import('../lib/api')
        const s = await getModelStatus(data.job_id)
        if (s.status === 'complete' || s.status === 'failed') {
          clearInterval(timer)
          setJobStatus(s)
          setRunningJobId(null)
        }
      }, 3000)
    },
  })

  const addHoliday = () =>
    setHolidays((h) => [...h, { label: '', start_date: '', end_date: '' }])

  const removeHoliday = (i: number) => setHolidays((h) => h.filter((_, idx) => idx !== i))

  const updateHoliday = (i: number, field: keyof Holiday, value: string) =>
    setHolidays((h) => h.map((x, idx) => idx === i ? { ...x, [field]: value } : x))

  const handleRun = () => {
    if (!sessionId || !activeModelId) return
    runMutation.mutate({
      session_id: sessionId,
      base_model_run_id: activeModelId,
      holidays,
      seasonality,
    })
  }

  if (!sessionId) {
    return (
      <div className="p-8">
        <div className="card text-center text-gray-500">
          Please upload data and run a model first.
        </div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        title="Tune Model"
        subtitle="Add holidays and seasonality adjustments then re-run"
        actions={
          jobStatus?.status === 'complete' ? (
            <button className="btn-primary" onClick={() => navigate('/results')}>
              View Results
            </button>
          ) : null
        }
      />

      <div className="p-8 space-y-6 max-w-3xl">
        {/* Base model */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-2">Base model</h3>
          {baseRun ? (
            <div className="flex items-center gap-3 text-sm text-gray-600">
              <StatusBadge status={baseRun.status} />
              <span>{baseRun.name ?? `Model ${baseRun.model_num} Iter ${baseRun.iteration_num}`}</span>
              <span className="text-gray-400">MAPE: {baseRun.mape?.toFixed(2) ?? '—'}%</span>
            </div>
          ) : (
            <p className="text-sm text-gray-400">
              No active model selected. <a className="text-blue-600" href="/results">Select one in Results</a>.
            </p>
          )}
        </div>

        {/* Job status */}
        {jobStatus && (
          <div className="card flex items-center gap-4">
            <StatusBadge status={jobStatus.status} />
            <span className="text-sm text-gray-600">
              {jobStatus.status === 'running' && 'Re-running model with tuning config...'}
              {jobStatus.status === 'complete' && 'Tuned model complete!'}
              {jobStatus.status === 'failed' && 'Tuned model run failed.'}
            </span>
            {jobStatus.status === 'running' && <Spinner />}
          </div>
        )}

        {/* Seasonality */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Seasonality components</h3>
          <div className="flex flex-wrap gap-6">
            {(Object.keys(seasonality) as (keyof typeof seasonality)[]).map((k) => (
              <label key={k} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={seasonality[k]}
                  onChange={(e) => setSeasonality((s) => ({ ...s, [k]: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="text-sm text-gray-700 capitalize">{k.replace('_', '-')}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Holidays */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Holiday periods</h3>
            <button className="btn-secondary flex items-center gap-1.5" onClick={addHoliday}>
              <Plus size={14} />
              Add holiday
            </button>
          </div>

          {holidays.length === 0 && (
            <p className="text-sm text-gray-400">No holidays added. Click "Add holiday" to include holiday periods.</p>
          )}

          <div className="space-y-3">
            {holidays.map((h, i) => (
              <div key={i} className="grid grid-cols-3 gap-3 items-end">
                <div>
                  <label className="label">Label</label>
                  <input
                    className="input"
                    placeholder="e.g. Eid Al-Adha"
                    value={h.label}
                    onChange={(e) => updateHoliday(i, 'label', e.target.value)}
                  />
                </div>
                <div>
                  <label className="label">Start date</label>
                  <input
                    type="date"
                    className="input"
                    value={h.start_date}
                    onChange={(e) => updateHoliday(i, 'start_date', e.target.value)}
                  />
                </div>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="label">End date</label>
                    <input
                      type="date"
                      className="input"
                      value={h.end_date}
                      onChange={(e) => updateHoliday(i, 'end_date', e.target.value)}
                    />
                  </div>
                  <button className="text-red-400 hover:text-red-600 mt-6" onClick={() => removeHoliday(i)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <button
          className="btn-primary w-full py-3"
          onClick={handleRun}
          disabled={!baseRun || runMutation.isPending || jobStatus?.status === 'running'}
        >
          {runMutation.isPending || jobStatus?.status === 'running' ? (
            <span className="flex items-center justify-center gap-2"><Spinner />Re-running model...</span>
          ) : (
            'Re-run with Tuning'
          )}
        </button>

        {/* History */}
        {(history as any[]).length > 0 && (
          <div className="card">
            <button
              className="w-full flex items-center justify-between"
              onClick={() => setHistoryOpen(!historyOpen)}
            >
              <h3 className="font-semibold text-gray-900">Tuning history ({(history as any[]).length})</h3>
              <ChevronDown size={16} className={`transition-transform ${historyOpen ? 'rotate-180' : ''}`} />
            </button>
            {historyOpen && (
              <div className="mt-4 space-y-2 text-sm">
                {(history as any[]).map((tc: any) => (
                  <div key={tc.id} className="bg-gray-50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">{tc.created_at}</div>
                    <div className="text-gray-600">
                      Holidays: {tc.holidays.length} |
                      Seasonality: {Object.entries(tc.seasonality).filter(([, v]) => v).map(([k]) => k).join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
