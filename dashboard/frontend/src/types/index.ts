export interface HealthIndexResponse {
  files: string[];
  hi_proxy: number[];
  hi_cnn: number[];
  train_n: number;
  mae_train: number;
  mae_val: number;
  onset_threshold: number;
  fail_threshold: number;
}

export interface RulResponse {
  files: string[];
  health_index: number[];
  rul_estimate: number[];
  rul_lower_1sigma: number[];
  rul_upper_1sigma: number[];
  max_life: number;
}

export interface BpfoTrendResponse {
  file_idx: number[];
  bpfo_rate: number[];
  bpfi_rate: number[];
  bsf_rate: number[];
  ftf_rate: number[];
  mean_rms: number[];
}

export type FaultType = 'bpfo' | 'bpfi' | 'bsf' | 'ftf';

export interface FileDetailResponse {
  idx: number;
  file: string;
  hi_proxy: number;
  hi_cnn: number;
  rul_estimate: number;
  rul_lower_1sigma: number;
  rul_upper_1sigma: number;
  kurtosis: number;
  entropy: number;
  match_counts: Record<FaultType, number>;
  dominant_fault: FaultType;
  dominant_fault_label: string;
  dominant_fault_strength: number;
  dominant_fault_hz: number;
  fault_freqs_hz: Record<FaultType, number>;
}

export interface ScalogramIndexResponse {
  available_indices: number[];
}

/** Derived health stage for a given HI value, used for color-coding across the UI. */
export type HealthStage = 'healthy' | 'warning' | 'failure';

export function stageForHi(hi: number, onset: number, fail: number): HealthStage {
  if (hi < fail) return 'failure';
  if (hi < onset) return 'warning';
  return 'healthy';
}
