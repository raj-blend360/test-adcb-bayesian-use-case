import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, CheckCircle, AlertCircle } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import { uploadChannelCSV } from '../lib/api'
import { useStore } from '../store'

const CHANNEL_COLS = ['date', 'conversions', 'spends_channel1', 'media_impressions_channel1', 'media_clicks_channel1', 'exogenous_holiday_flag']

function DropZone({
  label,
  accept,
  onFile,
  status,
}: {
  label: string
  accept: string
  onFile: (f: File) => void
  status: 'idle' | 'loading' | 'success' | 'error'
}) {
  const [dragging, setDragging] = useState(false)

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) onFile(file)
    },
    [onFile],
  )

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
        dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onClick={() => document.getElementById(`file-${label}`)?.click()}
    >
      <input
        id={`file-${label}`}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
      />
      <div className="flex flex-col items-center gap-2">
        {status === 'loading' ? (
          <Spinner className="h-8 w-8" />
        ) : status === 'success' ? (
          <CheckCircle className="text-green-500 h-8 w-8" />
        ) : status === 'error' ? (
          <AlertCircle className="text-red-500 h-8 w-8" />
        ) : (
          <UploadIcon className="text-gray-400 h-8 w-8" />
        )}
        <p className="text-sm font-medium text-gray-700">{label}</p>
        <p className="text-xs text-gray-400">Drag & drop or click to browse</p>
      </div>
    </div>
  )
}

export default function UploadPage() {
  const navigate = useNavigate()
  const setUpload = useStore((s) => s.setUpload)
  const upload = useStore((s) => s.upload)

  const [channelPreview, setChannelPreview] = useState<any[]>([])
  const [channelCols, setChannelCols] = useState<string[]>([])
  const [channelStatus, setChannelStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)

  const handleChannelFile = async (file: File) => {
    setChannelStatus('loading')
    setError(null)
    try {
      const res = await uploadChannelCSV(file, upload.sessionId ?? undefined)
      setChannelPreview(res.preview)
      setChannelCols(res.columns)
      setUpload({
        sessionId: res.session_id,
        channels: res.preview.map((r: any) => r.channel).filter(Boolean),
      })
      setChannelStatus('success')
    } catch (e: any) {
      setError(e.response?.data?.detail ?? 'Upload failed')
      setChannelStatus('error')
    }
  }

  return (
    <div>
      <PageHeader
        title="Upload Data"
        subtitle="Upload a single weekly CSV file with wide channel columns"
        actions={
          channelStatus === 'success' ? (
            <button className="btn-primary" onClick={() => navigate('/transform')}>
              Continue to Transform
            </button>
          ) : null
        }
      />

      <div className="p-8 space-y-8 max-w-4xl">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Required columns hint */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-3">Expected columns (single CSV)</h3>
          <div className="flex flex-wrap gap-2">
            {CHANNEL_COLS.map((c) => (
              <span key={c} className="badge badge-blue">{c}</span>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Required: <strong>date, conversions, spends_&lt;channel&gt;</strong>. Optional per-channel: media_impressions_&lt;channel&gt;, media_clicks_&lt;channel&gt;. Optional global controls: exogenous_*.
          </p>
        </div>

        {/* Upload zones */}
        <div className="grid grid-cols-1 gap-6">
          <div className="space-y-2">
            <h3 className="font-medium text-gray-900">Weekly MMM data <span className="text-red-500">*</span></h3>
            <DropZone label="MMM CSV" accept=".csv" onFile={handleChannelFile} status={channelStatus} />
          </div>
        </div>

        {/* Preview */}
        {channelPreview.length > 0 && (
          <div className="card overflow-hidden">
            <h3 className="font-semibold text-gray-900 mb-4">Data preview (first 10 rows)</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    {channelCols.map((c) => (
                      <th key={c} className="px-3 py-2 text-left font-medium text-gray-600">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {channelPreview.map((row, i) => (
                    <tr key={i} className="border-b hover:bg-gray-50">
                      {channelCols.map((c) => (
                        <td key={c} className="px-3 py-2 text-gray-700">{String(row[c] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
