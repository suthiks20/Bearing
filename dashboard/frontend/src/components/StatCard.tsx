import { useEffect, useState, type ReactNode } from 'react';
import { motion, useSpring } from 'framer-motion';
import type { HealthStage } from '../types';

const ACCENT_TEXT: Record<HealthStage, string> = {
  healthy: 'text-healthy',
  warning: 'text-warning',
  failure: 'text-failure',
};

const ACCENT_GLOW: Record<HealthStage, string> = {
  healthy: 'shadow-glow',
  warning: 'shadow-[0_0_40px_rgba(245,158,11,0.2)]',
  failure: 'shadow-[0_0_40px_rgba(239,68,68,0.25)]',
};

/** Tracks a target number with spring physics and exposes the live interpolated value for display. */
function useSpringNumber(target: number): number {
  const spring = useSpring(target, { stiffness: 110, damping: 20, mass: 0.5 });
  const [display, setDisplay] = useState(target);

  useEffect(() => {
    spring.set(target);
  }, [target, spring]);

  useEffect(() => {
    const unsubscribe = spring.on('change', (v) => setDisplay(v));
    return unsubscribe;
  }, [spring]);

  return display;
}

export interface ThresholdMarker {
  value: number;
  color: string;
  label: string;
}

export interface StatCardProps {
  label: string;
  value: number;
  unit?: string;
  decimals?: number;
  accent?: HealthStage;
  icon?: ReactNode;
  sublabel?: string;
  /** When provided, renders a horizontal track (0-1 domain) with threshold tick marks. */
  barMarkers?: ThresholdMarker[];
}

export default function StatCard({
  label,
  value,
  unit,
  decimals = 2,
  accent = 'healthy',
  icon,
  sublabel,
  barMarkers,
}: StatCardProps) {
  const display = useSpringNumber(value);
  const barDisplay = useSpringNumber(barMarkers ? value : 0);

  return (
    <motion.div
      layout
      className={`glass-card p-5 flex flex-col gap-3 transition-shadow duration-500 ${ACCENT_GLOW[accent]}`}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-widest text-slate-400">
          {label}
        </span>
        {icon && <span className={ACCENT_TEXT[accent]}>{icon}</span>}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className={`font-mono tabular-num text-3xl font-semibold ${ACCENT_TEXT[accent]}`}>
          {display.toFixed(decimals)}
        </span>
        {unit && <span className="font-mono text-sm text-slate-500">{unit}</span>}
      </div>

      {sublabel && <p className="font-sans text-xs text-slate-500 -mt-1">{sublabel}</p>}

      {barMarkers && (
        <div className="relative mt-1 h-1.5 rounded-full bg-base-700 overflow-hidden">
          <motion.div
            className={`absolute inset-y-0 left-0 rounded-full ${
              accent === 'failure' ? 'bg-failure' : accent === 'warning' ? 'bg-warning' : 'bg-healthy'
            }`}
            style={{ width: `${Math.min(Math.max(barDisplay, 0), 1) * 100}%` }}
          />
          {barMarkers.map((m) => (
            <div
              key={m.label}
              className="absolute top-0 bottom-0 w-px"
              style={{ left: `${m.value * 100}%`, backgroundColor: m.color }}
              title={m.label}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
}
