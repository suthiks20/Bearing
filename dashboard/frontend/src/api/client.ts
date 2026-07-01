import type {
  BpfoTrendResponse,
  FileDetailResponse,
  HealthIndexResponse,
  RulResponse,
  ScalogramIndexResponse,
} from '../types';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Static data fetch failed: ${path} — ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// file-details.json is loaded once and reused for every fileDetail() call.
// This avoids 984 separate fetches while keeping the same async interface.
let _fileDetailsCache: FileDetailResponse[] | null = null;
async function getFileDetails(): Promise<FileDetailResponse[]> {
  if (!_fileDetailsCache) {
    _fileDetailsCache = await getJson<FileDetailResponse[]>('/data/file-details.json');
  }
  return _fileDetailsCache;
}

export const api = {
  healthIndex:    () => getJson<HealthIndexResponse>('/data/health-index.json'),
  rul:            () => getJson<RulResponse>('/data/rul.json'),
  bpfoTrend:      () => getJson<BpfoTrendResponse>('/data/bpfo-trend.json'),
  scalogramIndex: () => getJson<ScalogramIndexResponse>('/data/scalogram-index.json'),

  fileDetail: async (idx: number): Promise<FileDetailResponse> => {
    const details = await getFileDetails();
    const entry = details[idx];
    if (!entry) throw new Error(`No file detail for index ${idx} (array length ${details.length})`);
    return entry;
  },

  // Returns a direct path to the precomputed PNG — no backend needed.
  // ScalogramViewer already resolves the nearest available index before calling this.
  scalogramUrl: (idx: number): string =>
    `/data/scalograms/file_${String(idx).padStart(4, '0')}.png`,
};
