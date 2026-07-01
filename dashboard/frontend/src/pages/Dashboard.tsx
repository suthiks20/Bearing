import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Clock, Zap } from 'lucide-react';

import { api } from '../api/client';
import type {
  FileDetailResponse,
  HealthIndexResponse,
  RulResponse,
  ScalogramIndexResponse,
} from '../types';
import { stageForHi } from '../types';

import HeroIntro from '../components/HeroIntro';
import ParticleBackground from '../components/ParticleBackground';
import StatCard from '../components/StatCard';
import FileScrubber from '../components/FileScrubber';
import RULCurve, { type RULDataPoint } from '../components/RULCurve';
import HealthIndexTimeline, { type HIDataPoint } from '../components/HealthIndexTimeline';
import CNNvsProxyPanel, { type CNNDataPoint } from '../components/CNNvsProxyPanel';
import FaultFrequencyPanel from '../components/FaultFrequencyPanel';
import ScalogramViewer from '../components/ScalogramViewer';

const FILE_DETAIL_DEBOUNCE_MS = 120;

function LoadingOverlay() {
  return (
    <div className="flex items-center justify-center py-32 gap-3 text-slate-500 font-mono text-sm">
      <span className="animate-spin inline-block w-4 h-4 border-2 border-slate-600 border-t-healthy rounded-full" />
      Connecting to backend…
    </div>
  );
}

export default function Dashboard() {
  const [hiData, setHiData] = useState<HealthIndexResponse | null>(null);
  const [rulData, setRulData] = useState<RulResponse | null>(null);
  const [scalogramIndex, setScalogramIndex] = useState<ScalogramIndexResponse | null>(null);
  const [fileDetail, setFileDetail] = useState<FileDetailResponse | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    Promise.all([api.healthIndex(), api.rul(), api.scalogramIndex()])
      .then(([hi, rul, sci]) => {
        setHiData(hi);
        setRulData(rul);
        setScalogramIndex(sci);
      })
      .catch(console.error);
  }, []);

  const fetchDetail = useCallback((idx: number) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      api.fileDetail(idx).then(setFileDetail).catch(console.error);
    }, FILE_DETAIL_DEBOUNCE_MS);
  }, []);

  useEffect(() => {
    fetchDetail(currentIdx);
  }, [currentIdx, fetchDetail]);

  const handleScrub = useCallback(
    (idx: number) => {
      setCurrentIdx(idx);
    },
    [],
  );

  const rulChartData = useMemo<RULDataPoint[]>(() => {
    if (!rulData) return [];
    return rulData.rul_estimate.map((rul, i) => {
      const lo = rulData.rul_lower_1sigma[i];
      const hi = rulData.rul_upper_1sigma[i];
      return {
        idx: i,
        rul_estimate: rul,
        rul_lower: lo,
        rul_upper: hi,
        band_width: Math.max(0, hi - lo),
      };
    });
  }, [rulData]);

  const hiChartData = useMemo<HIDataPoint[]>(() => {
    if (!hiData) return [];
    return hiData.hi_proxy.map((hi, i) => ({ idx: i, hi_proxy: hi }));
  }, [hiData]);

  const cnnChartData = useMemo<CNNDataPoint[]>(() => {
    if (!hiData) return [];
    return hiData.hi_proxy.map((hi, i) => ({
      idx: i,
      hi_proxy: hi,
      hi_cnn: hiData.hi_cnn[i],
    }));
  }, [hiData]);

  const onset = hiData?.onset_threshold ?? 0.7;
  const fail = hiData?.fail_threshold ?? 0.5;
  const hiProxy = fileDetail?.hi_proxy ?? 1;
  const hiStage = stageForHi(hiProxy, onset, fail);
  const rul = fileDetail?.rul_estimate ?? 0;
  const rulStage = rul < 20 ? 'failure' : rul < 100 ? 'warning' : 'healthy';

  // Bearing glow from match counts (normalize to 0-1 against max reasonable = 6 hits)
  const bpfoIntensity = Math.min((fileDetail?.match_counts?.bpfo ?? 0) / 6, 1);
  const bpfiIntensity = Math.min((fileDetail?.match_counts?.bpfi ?? 0) / 6, 1);
  // severity from HI (inverted: lower HI = higher severity)
  const severity = Math.max(0, 1 - hiProxy);

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <ParticleBackground />
      </div>

      <HeroIntro
        bpfoIntensity={bpfoIntensity}
        bpfiIntensity={bpfiIntensity}
        severity={severity}
      >
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-12 space-y-6">
          {!hiData && <LoadingOverlay />}
          {/* FileScrubber */}
          {hiData && (
            <FileScrubber
              files={hiData.files}
              healthIndex={hiData.hi_proxy}
              onsetThreshold={onset}
              failThreshold={fail}
              value={currentIdx}
              onChange={handleScrub}
            />
          )}

          {/* Stat cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatCard
              label="RUL Estimate"
              value={rul}
              unit="files"
              decimals={1}
              accent={rulStage}
              icon={<Clock size={16} />}
              sublabel={`file ${currentIdx} / ${(rulData?.max_life ?? 983)}`}
            />
            <StatCard
              label="Health Index"
              value={hiProxy}
              decimals={4}
              accent={hiStage}
              icon={<Activity size={16} />}
              barMarkers={[
                { value: fail, color: '#ef4444', label: 'failure' },
                { value: onset, color: '#f59e0b', label: 'onset' },
              ]}
            />
            <StatCard
              label="Dominant Fault"
              value={fileDetail?.dominant_fault_hz ?? 0}
              unit="Hz"
              decimals={2}
              accent={hiStage}
              icon={<Zap size={16} />}
              sublabel={fileDetail?.dominant_fault_label ?? '—'}
            />
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <RULCurve
              data={rulChartData}
              currentIdx={currentIdx}
              maxLife={rulData?.max_life ?? 983}
            />
            <HealthIndexTimeline
              data={hiChartData}
              currentIdx={currentIdx}
              onsetThreshold={onset}
              failThreshold={fail}
            />
          </div>

          {/* Scalogram + Fault frequency row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ScalogramViewer
              currentIdx={currentIdx}
              availableIndices={scalogramIndex?.available_indices ?? []}
            />
            {fileDetail && <FaultFrequencyPanel detail={fileDetail} />}
          </div>

          {/* CNN vs Proxy honest section */}
          {hiData && (
            <CNNvsProxyPanel
              data={cnnChartData}
              currentIdx={currentIdx}
              trainN={hiData.train_n}
              maeTrain={hiData.mae_train}
              maeVal={hiData.mae_val}
            />
          )}
        </div>
      </HeroIntro>
    </div>
  );
}
