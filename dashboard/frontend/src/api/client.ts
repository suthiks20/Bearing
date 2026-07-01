import type {
  BpfoTrendResponse,
  FileDetailResponse,
  HealthIndexResponse,
  RulResponse,
  ScalogramIndexResponse,
} from '../types';

const BASE_URL = 'http://localhost:8000/api';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  healthIndex: () => getJson<HealthIndexResponse>('/health-index'),
  rul: () => getJson<RulResponse>('/rul'),
  bpfoTrend: () => getJson<BpfoTrendResponse>('/bpfo-trend'),
  fileDetail: (idx: number) => getJson<FileDetailResponse>(`/file/${idx}`),
  scalogramIndex: () => getJson<ScalogramIndexResponse>('/scalogram-index'),
  scalogramUrl: (idx: number) => `${BASE_URL}/scalogram/${idx}`,
};
