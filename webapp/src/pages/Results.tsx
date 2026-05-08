import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Eye, Save } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import { listResults, saveModelName } from '../lib/api'
import { useStore } from '../store'
import { fmt, fmtPct } from '../lib/utils'

export default function ResultsPage() {
  const navigate = useNavigate()
  const sessionId = useStore((s) => s.upload.sessionId)
  const setActiveModelId = useStore((s) => s.setActiveModelId)
  const activeModelId = useStore((s) => s.activeModelId)
  const qc = useQueryClient()

  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [savingId, setSavingId] = useState<number | null>(null)
  const [saveName, setSaveName] = useState('')

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['results', sessionId],
    queryFn: () => listResults(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (q) => {
      const data = q.state.data as any[]
      const hasPending = data?.some((r) => r.status === 'pending' || r.status === 'running')
      return hasPending ? 3000 : false
    },
  })

  const saveMutation = useMutation({
    mutationFn: ({ id, name }: { id: number; name: string }) => saveModelName(id, name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['results', sessionId] })
      setSavingId(null)
      setSaveName('')
    },
  })

  const toggleSelect = (id: number) => {
    setSelectedIds((s) => {
      const n = new Set(s)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })
  }

  const compareData = runs
    .filter((r: any) => selectedIds.has(r.id) && r.status === 'complete')
    .map((r: any) => ({
      name: r.name ?? `Model ${r.model_num} Iter ${r.iteration_num}`,
      MAPE: r.mape ?? 0,
      AdjR2: r.adj_r2 ?? 0,
    }))

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
        title="Model Results"
        subtitle="Compare model runs and save the best model"
        actions={
          activeModelId ? (
            <button className="btn-primary" onClick={() => navigate('/visualize')}>
              Visualize
            </button>
          ) : null
        }
      />

      <div className="p-8 space-y-6">
        {isLoading && <div className="text-center text-gray-500 py-8">Loading...</div>}

        {/* Results table */}
        <div className="card overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="px-4 py-3 text-left">Compare</th>
                  <th className="px-4 py-3 text-left">Model #</th>
                  <th className="px-4 py-3 text-left">Iter #</th>
                  <th className="px-4 py-3 text-left">Name</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-right">Adj R²</th>
                  <th className="px-4 py-3 text-right">MAPE</th>
                  <th className="px-4 py-3 text-right">R-hat pass</th>
                  <th className="px-4 py-3 text-right">CI Width</th>
                  <th className="px-4 py-3 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(runs as any[]).map((run) => (
                  <tr
                    key={run.id}
                    className={`border-b hover:bg-gray-50 ${activeModelId === run.id ? 'bg-blue-50' : ''}`}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedIds.has(run.id)}
                        onChange={() => toggleSelect(run.id)}
                      />
                    </td>
                    <td className="px-4 py-3 font-medium">{run.model_num}</td>
                    <td className="px-4 py-3">{run.iteration_num}</td>
                    <td className="px-4 py-3 text-gray-600">{run.name ?? <span className="text-gray-400">—</span>}</td>
                    <td className="px-4 py-3"><StatusBadge status={run.status} /></td>
                    <td className="px-4 py-3 text-right font-mono">{fmt(run.adj_r2, 3)}</td>
                    <td className="px-4 py-3 text-right font-mono">{run.mape != null ? `${fmt(run.mape, 2)}%` : '—'}</td>
                    <td className="px-4 py-3 text-right font-mono">{run.rhat_pass_pct != null ? `${fmt(run.rhat_pass_pct, 1)}%` : '—'}</td>
                    <td className="px-4 py-3 text-right font-mono">{fmt(run.confidence_width, 3)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <button
                          className="text-blue-600 hover:text-blue-800"
                          title="Set as active / visualize"
                          onClick={() => { setActiveModelId(run.id); navigate('/visualize') }}
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          className="text-gray-500 hover:text-gray-700"
                          title="Save name"
                          onClick={() => { setSavingId(run.id); setSaveName(run.name ?? '') }}
                        >
                          <Save size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Save name dialog */}
        {savingId && (
          <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl p-6 w-80 shadow-xl space-y-4">
              <h3 className="font-semibold">Save model name</h3>
              <input
                className="input"
                placeholder="e.g. Baseline MAP"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                autoFocus
              />
              <div className="flex gap-3 justify-end">
                <button className="btn-secondary" onClick={() => setSavingId(null)}>Cancel</button>
                <button
                  className="btn-primary"
                  onClick={() => saveMutation.mutate({ id: savingId, name: saveName })}
                  disabled={!saveName.trim()}
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Comparison chart */}
        {compareData.length > 1 && (
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Model comparison</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={compareData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="MAPE" fill="#ef4444" />
                <Bar dataKey="AdjR2" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
