import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { Download } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import {
  getContributions, getResponseCurves, getWeeklyDecomp,
  getRoi, getWaterfall, getChartCsvUrl, listResults,
} from '../lib/api'
import { useStore } from '../store'
import { CHANNEL_COLORS } from '../lib/utils'

const TABS = ['Contributions', 'Response Curves', 'Weekly Decomp', 'ROI', 'Waterfall'] as const
type Tab = typeof TABS[number]

function downloadCsv(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}

export default function VisualizePage() {
  const sessionId = useStore((s) => s.upload.sessionId)
  const activeModelId = useStore((s) => s.activeModelId)
  const setActiveModelId = useStore((s) => s.setActiveModelId)
  const [activeTab, setActiveTab] = useState<Tab>('Contributions')

  const { data: runs = [] } = useQuery({
    queryKey: ['results', sessionId],
    queryFn: () => listResults(sessionId!),
    enabled: !!sessionId,
  })
  const completeRuns = (runs as any[]).filter((r: any) => r.status === 'complete')

  const modelId = activeModelId ?? completeRuns[0]?.id

  const { data: contribs, isLoading: loadingC } = useQuery({
    queryKey: ['contributions', modelId],
    queryFn: () => getContributions(modelId!),
    enabled: !!modelId && activeTab === 'Contributions',
  })

  const { data: curves, isLoading: loadingCurves } = useQuery({
    queryKey: ['response_curves', modelId],
    queryFn: () => getResponseCurves(modelId!),
    enabled: !!modelId && activeTab === 'Response Curves',
  })

  const { data: weekly, isLoading: loadingW } = useQuery({
    queryKey: ['weekly', modelId],
    queryFn: () => getWeeklyDecomp(modelId!),
    enabled: !!modelId && activeTab === 'Weekly Decomp',
  })

  const { data: roiData, isLoading: loadingRoi } = useQuery({
    queryKey: ['roi', modelId],
    queryFn: () => getRoi(modelId!),
    enabled: !!modelId && activeTab === 'ROI',
  })

  const { data: waterfallData, isLoading: loadingWf } = useQuery({
    queryKey: ['waterfall', modelId],
    queryFn: () => getWaterfall(modelId!),
    enabled: !!modelId && activeTab === 'Waterfall',
  })

  const isLoading = loadingC || loadingCurves || loadingW || loadingRoi || loadingWf

  if (!sessionId) {
    return (
      <div className="p-8">
        <div className="card text-center text-gray-500">Please upload data and run a model first.</div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader title="Visualize" subtitle="Explore model outputs and download charts" />

      <div className="p-8 space-y-6">
        {/* Model selector */}
        <div className="flex items-center gap-4">
          <label className="text-sm font-medium text-gray-700">Model</label>
          <select
            className="input w-64"
            value={modelId ?? ''}
            onChange={(e) => setActiveModelId(Number(e.target.value))}
          >
            {completeRuns.map((r: any) => (
              <option key={r.id} value={r.id}>
                {r.name ?? `Model ${r.model_num} Iter ${r.iteration_num}`}
                {r.mape != null ? ` (MAPE ${r.mape.toFixed(2)}%)` : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1 w-fit">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="card">
          {isLoading && (
            <div className="flex items-center justify-center py-12">
              <Spinner className="h-8 w-8" />
            </div>
          )}

          {/* Contributions */}
          {activeTab === 'Contributions' && !loadingC && contribs && (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold">Channel Contributions</h3>
                <button
                  className="btn-secondary flex items-center gap-1.5"
                  onClick={() => downloadCsv(getChartCsvUrl(modelId!, 'contributions'), 'contributions.csv')}
                >
                  <Download size={14} /> CSV
                </button>
              </div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={contribs.data ?? []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="contribution" name="Contribution">
                    {(contribs.data ?? []).map((_: any, i: number) => (
                      <Cell key={i} fill={CHANNEL_COLORS[i % CHANNEL_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Response Curves */}
          {activeTab === 'Response Curves' && !loadingCurves && curves && (
            <div>
              <h3 className="font-semibold mb-4">Response Curves per Channel</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(curves.curves ?? {}).map(([ch, data]: [string, any], i) => (
                  <div key={ch}>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">{ch}</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={(data.x as number[]).map((x: number, j: number) => ({
                        spend: Math.round(x),
                        mean: data.y_mean[j],
                        lower: data.y_lower[j],
                        upper: data.y_upper[j],
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="spend" tick={{ fontSize: 9 }} />
                        <YAxis tick={{ fontSize: 9 }} />
                        <Tooltip />
                        <Area type="monotone" dataKey="upper" stroke="none" fill={CHANNEL_COLORS[i % CHANNEL_COLORS.length]} fillOpacity={0.2} />
                        <Area type="monotone" dataKey="mean" stroke={CHANNEL_COLORS[i % CHANNEL_COLORS.length]} fill="none" strokeWidth={2} />
                        <Area type="monotone" dataKey="lower" stroke="none" fill="white" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Weekly Decomp */}
          {activeTab === 'Weekly Decomp' && !loadingW && weekly && (
            <div>
              <h3 className="font-semibold mb-4">Weekly Media vs Non-Media</h3>
              {weekly.dates?.length > 0 ? (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={(weekly.dates as string[]).map((d: string, i: number) => ({
                    date: d,
                    media: weekly.media[i],
                    non_media: weekly.non_media[i],
                    actual: weekly.actual[i],
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="media" stackId="1" fill="#3b82f6" stroke="#3b82f6" name="Media" />
                    <Area type="monotone" dataKey="non_media" stackId="1" fill="#10b981" stroke="#10b981" name="Non-media" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-gray-400 text-center py-8">Weekly decomp data not available for this model run.</p>
              )}
            </div>
          )}

          {/* ROI */}
          {activeTab === 'ROI' && !loadingRoi && roiData && (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold">ROI by Channel</h3>
                <button
                  className="btn-secondary flex items-center gap-1.5"
                  onClick={() => downloadCsv(getChartCsvUrl(modelId!, 'roi'), 'roi.csv')}
                >
                  <Download size={14} /> CSV
                </button>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={roiData.roi ?? []} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={80} />
                  <Tooltip />
                  <Bar dataKey="roi" name="ROI">
                    {(roiData.roi ?? []).map((_: any, i: number) => (
                      <Cell key={i} fill={CHANNEL_COLORS[i % CHANNEL_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Waterfall */}
          {activeTab === 'Waterfall' && !loadingWf && waterfallData && (
            <div>
              <h3 className="font-semibold mb-4">Contribution Waterfall</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={waterfallData.waterfall ?? []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="running" name="Cumulative" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
