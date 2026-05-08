import axios from 'axios'

export const apiClient = axios.create({
  baseURL: '',
  timeout: 300000,
})

// ─── Upload ──────────────────────────────────────────────────────────────────

export async function uploadChannelCSV(file: File, sessionId?: number) {
  const form = new FormData()
  form.append('file', file)
  if (sessionId) form.append('session_id', String(sessionId))
  const { data } = await apiClient.post('/upload/channel', form)
  return data
}

export async function uploadCampaignCSV(file: File, sessionId: number) {
  const form = new FormData()
  form.append('file', file)
  form.append('session_id', String(sessionId))
  const { data } = await apiClient.post('/upload/campaign', form)
  return data
}

export async function getUploadColumns(sessionId: number) {
  const { data } = await apiClient.get(`/upload/columns/${sessionId}`)
  return data
}

// ─── Transform ───────────────────────────────────────────────────────────────

export async function saveTransformConfig(config: object) {
  const { data } = await apiClient.post('/transform/config', config)
  return data
}

export async function getTransformPreview(sessionId: number) {
  const { data } = await apiClient.get(`/transform/preview/${sessionId}`)
  return data
}

// ─── Model ───────────────────────────────────────────────────────────────────

export async function fitModel(payload: object) {
  const { data } = await apiClient.post('/model/fit', payload)
  return data
}

export async function getModelStatus(jobId: number) {
  const { data } = await apiClient.get(`/model/status/${jobId}`)
  return data
}

// ─── Results ─────────────────────────────────────────────────────────────────

export async function listResults(sessionId: number) {
  const { data } = await apiClient.get('/results', { params: { session_id: sessionId } })
  return data
}

export async function getResult(modelId: number) {
  const { data } = await apiClient.get(`/results/${modelId}`)
  return data
}

export async function saveModelName(modelId: number, name: string) {
  const { data } = await apiClient.post(`/results/${modelId}/save`, { name })
  return data
}

// ─── Tune ────────────────────────────────────────────────────────────────────

export async function saveTuneConfig(config: object) {
  const { data } = await apiClient.post('/tune/config', config)
  return data
}

export async function runTunedModel(config: object) {
  const { data } = await apiClient.post('/tune/run', config)
  return data
}

export async function getTuneHistory(sessionId: number) {
  const { data } = await apiClient.get(`/tune/history/${sessionId}`)
  return data
}

// ─── Visualize ───────────────────────────────────────────────────────────────

export async function getContributions(modelId: number) {
  const { data } = await apiClient.get(`/visualize/${modelId}/contributions`)
  return data
}

export async function getResponseCurves(modelId: number) {
  const { data } = await apiClient.get(`/visualize/${modelId}/response_curves`)
  return data
}

export async function getWeeklyDecomp(modelId: number) {
  const { data } = await apiClient.get(`/visualize/${modelId}/weekly`)
  return data
}

export async function getRoi(modelId: number) {
  const { data } = await apiClient.get(`/visualize/${modelId}/roi`)
  return data
}

export async function getWaterfall(modelId: number) {
  const { data } = await apiClient.get(`/visualize/${modelId}/waterfall`)
  return data
}

export function getChartCsvUrl(modelId: number, chartType: string) {
  return `/visualize/${modelId}/${chartType}/csv`
}

// ─── Optimize ────────────────────────────────────────────────────────────────

export async function forwardOptimize(payload: object) {
  const { data } = await apiClient.post('/optimize/forward', payload)
  return data
}

export async function reverseOptimize(payload: object) {
  const { data } = await apiClient.post('/optimize/reverse', payload)
  return data
}

export async function listScenarios(sessionId: number) {
  const { data } = await apiClient.get(`/optimize/scenarios/${sessionId}`)
  return data
}
