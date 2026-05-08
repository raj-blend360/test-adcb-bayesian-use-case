import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UploadState {
  sessionId: number | null
  channels: string[]
  campaigns: string[]
  hasCampaignData: boolean
}

interface TransformConfig {
  include_seasonality: boolean
  include_holiday: boolean
  include_promo: boolean
  seasonality_periods: number[]
  n_harmonics: number
  test_weeks: number
  channels: ChannelConfig[]
}

interface ChannelConfig {
  channel: string
  adstock: { adstock_type: string; max_lag: number; decay_prior_mean: number }
  saturation: { saturation_type: string; alpha_prior_mean: number; lambda_prior_mean: number }
  metric: string
}

interface AppState {
  // Upload
  upload: UploadState
  setUpload: (u: Partial<UploadState>) => void

  // Transform config
  transformConfig: TransformConfig | null
  setTransformConfig: (c: TransformConfig) => void

  // Active model
  activeModelId: number | null
  setActiveModelId: (id: number | null) => void

  // Running job
  runningJobId: number | null
  setRunningJobId: (id: number | null) => void
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      upload: { sessionId: null, channels: [], campaigns: [], hasCampaignData: false },
      setUpload: (u) => set((s) => ({ upload: { ...s.upload, ...u } })),

      transformConfig: null,
      setTransformConfig: (c) => set({ transformConfig: c }),

      activeModelId: null,
      setActiveModelId: (id) => set({ activeModelId: id }),

      runningJobId: null,
      setRunningJobId: (id) => set({ runningJobId: id }),
    }),
    { name: 'mmm-store' },
  ),
)
