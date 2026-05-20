import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import { getUploadColumns, saveTransformConfig, getTransformPreview } from '../lib/api'
import { useStore } from '../store'
import { CHANNEL_COLORS } from '../lib/utils'

const ADSTOCK_TYPES = ['geometric', 'weibull']
const SAT_TYPES = ['hill', 'logistic', 'michaelis_menten']

function defaultChannelConfig(channel: string) {
  return {
    channel,
    adstock: { adstock_type: 'geometric', max_lag: 4, decay_prior_mean: 0.5 },
    saturation: { saturation_type: 'hill', alpha_prior_mean: 2.0, lambda_prior_mean: 0.5 },
    metric: 'clicks',
  }
}

export default function TransformPage() {
  const navigate = useNavigate()
  const sessionId = useStore((s) => s.upload.sessionId)
  const setTransformConfig = useStore((s) => s.setTransformConfig)

  const [includeSeas, setIncludeSeas] = useState(true)
  const [includeHoliday, setIncludeHoliday] = useState(true)
  const [includePromo, setIncludePromo] = useState(false)
  const [testWeeks, setTestWeeks] = useState(12)
  const [randomHoldout, setRandomHoldout] = useState(false)
  const [channelConfigs, setChannelConfigs] = useState<ReturnType<typeof defaultChannelConfig>[]>([])
  const [previewData, setPreviewData] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [openChannel, setOpenChannel] = useState<string | null>(null)

  const { data: colData } = useQuery({
    queryKey: ['columns', sessionId],
    queryFn: () => getUploadColumns(sessionId!),
    enabled: !!sessionId,
  })
  const availableMetrics = ['media_spend', 'impressions', 'clicks'].filter((m) =>
    (colData?.channel_columns ?? []).includes(m))

  useEffect(() => {
    if (colData?.channels) {
      const defaultMetric = availableMetrics.includes('clicks')
        ? 'clicks'
        : (availableMetrics[0] ?? 'media_spend')
      setChannelConfigs(colData.channels.map((channel: string) => ({
        ...defaultChannelConfig(channel),
        metric: defaultMetric,
      })))
    }
  }, [colData, availableMetrics])

  const saveMutation = useMutation({
    mutationFn: (config: any) => saveTransformConfig(config),
    onSuccess: () => navigate('/model'),
  })

  const handlePreview = async () => {
    if (!sessionId) return
    setPreviewLoading(true)
    try {
      await saveTransformConfig(buildConfig())
      const data = await getTransformPreview(sessionId)
      setPreviewData(data)
    } finally {
      setPreviewLoading(false)
    }
  }

  function buildConfig() {
    return {
      session_id: sessionId,
      include_seasonality: includeSeas,
      include_holiday: includeHoliday,
      include_promo: includePromo,
      seasonality_periods: [52.0, 26.0],
      n_harmonics: 2,
      test_weeks: testWeeks,
      random_holdout: randomHoldout,
      holdout_seed: 42,
      channels: channelConfigs,
    }
  }

  function updateChannel(channel: string, field: string, subfield: string, value: any) {
    setChannelConfigs((prev) =>
      prev.map((c) =>
        c.channel === channel
          ? { ...c, [field]: { ...(c as any)[field], [subfield]: value } }
          : c,
      ),
    )
  }

  const previewChartData = previewData
    ? (() => {
        const firstCh = Object.keys(previewData)[0]
        if (!firstCh) return []
        const dates = previewData[firstCh].dates as string[]
        return dates.map((d: string, i: number) => ({
          date: d,
          ...Object.fromEntries(
            Object.entries(previewData).map(([ch, v]: [string, any]) => [
              `${ch}_raw`, v.raw_spend[i],
            ]),
          ),
          ...Object.fromEntries(
            Object.entries(previewData).map(([ch, v]: [string, any]) => [
              `${ch}_ad`, v.adstocked_spend[i],
            ]),
          ),
        }))
      })()
    : []

  if (!sessionId) {
    return (
      <div className="p-8">
        <div className="card text-center text-gray-500">
          Please upload data first. <a className="text-blue-600" href="/upload">Go to Upload</a>
        </div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        title="Configure Transforms"
        subtitle="Set adstock and saturation parameters per channel"
        actions={
          <div className="flex gap-3">
            <button className="btn-secondary" onClick={handlePreview} disabled={previewLoading}>
              {previewLoading ? <Spinner /> : 'Preview Adstock'}
            </button>
            <button
              className="btn-primary"
              onClick={() => {
                const cfg = buildConfig()
                setTransformConfig(cfg as any)
                saveMutation.mutate(cfg)
              }}
              disabled={saveMutation.isPending}
            >
              {saveMutation.isPending ? <Spinner /> : 'Save & Continue'}
            </button>
          </div>
        }
      />

      <div className="p-8 space-y-6 max-w-5xl">
        {/* Global features */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Feature selection</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Annual seasonality', value: includeSeas, set: setIncludeSeas },
              { label: 'Holiday flag', value: includeHoliday, set: setIncludeHoliday },
              { label: 'Promo flag', value: includePromo, set: setIncludePromo },
            ].map(({ label, value, set }) => (
              <label key={label} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={value}
                  onChange={(e) => set(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600"
                />
                <span className="text-sm text-gray-700">{label}</span>
              </label>
            ))}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={randomHoldout}
                onChange={(e) => setRandomHoldout(e.target.checked)}
                className="rounded border-gray-300 text-blue-600"
              />
              <span className="text-sm text-gray-700">Random holdout periods</span>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-sm text-gray-700">Test weeks</span>
              <input
                type="number"
                value={testWeeks}
                onChange={(e) => setTestWeeks(Number(e.target.value))}
                className="input w-20"
                min={4}
                max={52}
              />
            </label>
          </div>
        </div>

        {/* Per-channel config */}
        <div className="space-y-3">
          {channelConfigs.map((cfg, idx) => (
            <div key={cfg.channel} className="card">
              <button
                className="w-full flex items-center justify-between"
                onClick={() => setOpenChannel(openChannel === cfg.channel ? null : cfg.channel)}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: CHANNEL_COLORS[idx % CHANNEL_COLORS.length] }}
                  />
                  <span className="font-medium text-gray-900">{cfg.channel}</span>
                </div>
                <span className="text-xs text-gray-400">
                  {cfg.adstock.adstock_type} / lag {cfg.adstock.max_lag} | {cfg.saturation.saturation_type}
                </span>
              </button>

              {openChannel === cfg.channel && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t">
                  {/* Adstock */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-medium text-gray-800">Adstock</h4>
                    <div>
                      <label className="label">Type</label>
                      <select
                        className="input"
                        value={cfg.adstock.adstock_type}
                        onChange={(e) => updateChannel(cfg.channel, 'adstock', 'adstock_type', e.target.value)}
                      >
                        {ADSTOCK_TYPES.map((t) => <option key={t}>{t}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="label">Max lag (weeks): {cfg.adstock.max_lag}</label>
                      <input
                        type="range"
                        min={1}
                        max={8}
                        value={cfg.adstock.max_lag}
                        onChange={(e) => updateChannel(cfg.channel, 'adstock', 'max_lag', Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="label">Decay prior mean: {cfg.adstock.decay_prior_mean}</label>
                      <input
                        type="range"
                        min={0.1}
                        max={0.9}
                        step={0.05}
                        value={cfg.adstock.decay_prior_mean}
                        onChange={(e) => updateChannel(cfg.channel, 'adstock', 'decay_prior_mean', Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </div>

                  {/* Saturation */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-medium text-gray-800">Saturation</h4>
                    <div>
                      <label className="label">Type</label>
                      <select
                        className="input"
                        value={cfg.saturation.saturation_type}
                        onChange={(e) => updateChannel(cfg.channel, 'saturation', 'saturation_type', e.target.value)}
                      >
                        {SAT_TYPES.map((t) => <option key={t}>{t}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="label">Alpha prior mean</label>
                      <input
                        type="number"
                        className="input"
                        step={0.1}
                        value={cfg.saturation.alpha_prior_mean}
                        onChange={(e) => updateChannel(cfg.channel, 'saturation', 'alpha_prior_mean', Number(e.target.value))}
                      />
                    </div>
                    <div>
                      <label className="label">Lambda prior mean</label>
                      <input
                        type="number"
                        className="input"
                        step={0.05}
                        value={cfg.saturation.lambda_prior_mean}
                        onChange={(e) => updateChannel(cfg.channel, 'saturation', 'lambda_prior_mean', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* Metric */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-medium text-gray-800">Input metric</h4>
                    {availableMetrics.map((m) => (
                      <label key={m} className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name={`metric-${cfg.channel}`}
                          value={m}
                          checked={cfg.metric === m}
                          onChange={() => setChannelConfigs((prev) =>
                            prev.map((c) => c.channel === cfg.channel ? { ...c, metric: m } : c),
                          )}
                        />
                        <span className="text-sm text-gray-700 capitalize">{m}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Preview chart */}
        {previewChartData.length > 0 && (
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Adstocked spend preview</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={previewChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                {Object.keys(previewData ?? {}).map((ch, i) => (
                  <Line
                    key={`${ch}_ad`}
                    dataKey={`${ch}_ad`}
                    name={`${ch} (adstocked)`}
                    stroke={CHANNEL_COLORS[i % CHANNEL_COLORS.length]}
                    dot={false}
                    strokeWidth={2}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
