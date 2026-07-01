import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../api/client';

interface ScalogramViewerProps {
  currentIdx: number;
  availableIndices: number[];
}

function nearestAvailable(idx: number, available: number[]): number | null {
  if (available.length === 0) return null;
  let best = available[0];
  let bestDist = Math.abs(idx - available[0]);
  for (const a of available) {
    const d = Math.abs(idx - a);
    if (d < bestDist) { bestDist = d; best = a; }
  }
  return best;
}

export default function ScalogramViewer({ currentIdx, availableIndices }: ScalogramViewerProps) {
  const nearest = nearestAvailable(currentIdx, availableIndices);
  const url = nearest !== null ? api.scalogramUrl(nearest) : null;

  const [displayUrl, setDisplayUrl] = useState<string | null>(url);
  const [displayIdx, setDisplayIdx] = useState<number | null>(nearest);
  const lastLoadedRef = useRef<string | null>(null);

  useEffect(() => {
    if (!url || url === lastLoadedRef.current) return;
    const img = new Image();
    img.onload = () => {
      lastLoadedRef.current = url;
      setDisplayUrl(url);
      setDisplayIdx(nearest);
    };
    img.src = url;
  }, [url, nearest]);

  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-widest text-slate-400">
          FrSST Scalogram
        </span>
        {displayIdx !== null && (
          <span className="font-mono text-[10px] text-slate-500">
            nearest cached: file {displayIdx}
          </span>
        )}
      </div>

      <div className="relative w-full aspect-[2/1] rounded-xl overflow-hidden bg-base-900">
        <AnimatePresence mode="sync">
          {displayUrl ? (
            <motion.img
              key={displayUrl}
              src={displayUrl}
              alt={`FrSST scalogram file ${displayIdx}`}
              className="absolute inset-0 w-full h-full object-contain"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            />
          ) : (
            <motion.div
              key="placeholder"
              className="absolute inset-0 flex items-center justify-center text-slate-600 font-mono text-xs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {availableIndices.length === 0 ? 'scalogram cache empty' : 'loading…'}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
