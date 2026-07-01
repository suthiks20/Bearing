import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Pause, Play } from 'lucide-react';
import { stageForHi } from '../types';

const STAGE_COLOR: Record<'healthy' | 'warning' | 'failure', string> = {
  healthy: '#22d3ee',
  warning: '#f59e0b',
  failure: '#ef4444',
};

const AUTOPLAY_INTERVAL_MS = 70;

export interface FileScrubberProps {
  files: string[];
  healthIndex: number[];
  onsetThreshold: number;
  failThreshold: number;
  value: number;
  onChange: (idx: number) => void;
}

export default function FileScrubber({
  files,
  healthIndex,
  onsetThreshold,
  failThreshold,
  value,
  onChange,
}: FileScrubberProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const n = files.length;

  const gradient = useMemo(() => {
    if (n === 0) return 'transparent';
    const stops: string[] = [];
    const step = Math.max(1, Math.floor(n / 240));
    for (let i = 0; i < n; i += step) {
      const stage = stageForHi(healthIndex[i], onsetThreshold, failThreshold);
      const pct = (i / (n - 1)) * 100;
      stops.push(`${STAGE_COLOR[stage]} ${pct.toFixed(2)}%`);
    }
    const lastStage = stageForHi(healthIndex[n - 1], onsetThreshold, failThreshold);
    stops.push(`${STAGE_COLOR[lastStage]} 100%`);
    return `linear-gradient(to right, ${stops.join(', ')})`;
  }, [n, healthIndex, onsetThreshold, failThreshold]);

  const indexFromClientX = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      if (!track || n === 0) return 0;
      const rect = track.getBoundingClientRect();
      const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
      return Math.round(ratio * (n - 1));
    },
    [n],
  );

  useEffect(() => {
    if (!dragging) return;
    const handleMove = (e: PointerEvent) => {
      const idx = indexFromClientX(e.clientX);
      setHoverIdx(idx);
      onChange(idx);
    };
    const handleUp = () => setDragging(false);
    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };
  }, [dragging, indexFromClientX, onChange]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      onChange(value >= n - 1 ? 0 : value + 1);
    }, AUTOPLAY_INTERVAL_MS);
    return () => clearInterval(id);
  }, [playing, value, n, onChange]);

  const pct = n > 1 ? (value / (n - 1)) * 100 : 0;
  const tooltipIdx = dragging ? (hoverIdx ?? value) : value;

  return (
    <div className="glass-card p-5 flex items-center gap-4">
      <button
        type="button"
        onClick={() => setPlaying((p) => !p)}
        className="shrink-0 grid place-items-center w-10 h-10 rounded-full border border-white/10 bg-white/[0.04] text-healthy hover:bg-white/[0.08] transition-colors"
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
      </button>

      <div className="relative flex-1 select-none">
        <div
          ref={trackRef}
          className="relative h-2.5 rounded-full cursor-pointer"
          style={{ background: gradient }}
          onPointerDown={(e) => {
            setDragging(true);
            onChange(indexFromClientX(e.clientX));
          }}
        >
          <motion.div
            className="absolute top-1/2 w-4 h-4 rounded-full bg-white shadow-[0_0_12px_rgba(255,255,255,0.6)] border border-base-950/40"
            style={{ left: `${pct}%` }}
            animate={{ left: `${pct}%`, y: '-50%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          />

          {dragging && (
            <div
              className="absolute -top-12 -translate-x-1/2 glass-card px-3 py-1.5 whitespace-nowrap pointer-events-none"
              style={{ left: `${pct}%` }}
            >
              <span className="font-mono text-xs text-slate-200">{files[tooltipIdx]}</span>
            </div>
          )}
        </div>

        <div className="flex justify-between mt-2 font-mono text-[10px] text-slate-500">
          <span>file {value} / {n - 1}</span>
          <span>{files[value]}</span>
        </div>
      </div>
    </div>
  );
}
