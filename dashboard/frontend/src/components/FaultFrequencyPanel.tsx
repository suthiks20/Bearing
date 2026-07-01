import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TooltipProps = { active?: boolean; payload?: any[] };
import type { FileDetailResponse, FaultType } from '../types';

const FAULT_COLORS: Record<FaultType, string> = {
  bpfo: '#f59e0b',
  bpfi: '#ef4444',
  bsf: '#22d3ee',
  ftf: '#a78bfa',
};

const FAULT_LABELS: Record<FaultType, string> = {
  bpfo: 'BPFO (outer)',
  bpfi: 'BPFI (inner)',
  bsf: 'BSF (ball)',
  ftf: 'FTF (cage)',
};

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0];
  return (
    <div className="glass-card px-3 py-2 text-xs font-mono">
      <span className="text-slate-400">{name} </span>
      <span className="text-white">{(value as number).toFixed(1)} hits</span>
    </div>
  );
}

interface FaultFrequencyPanelProps {
  detail: FileDetailResponse;
}

export default function FaultFrequencyPanel({ detail }: FaultFrequencyPanelProps) {
  const faults: FaultType[] = ['bpfo', 'bpfi', 'bsf', 'ftf'];
  const chartData = faults.map((f) => ({
    name: FAULT_LABELS[f],
    key: f,
    value: detail.match_counts[f] ?? 0,
    hz: detail.fault_freqs_hz[f],
  }));

  return (
    <div className="glass-card p-5 flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <span className="font-mono text-xs uppercase tracking-widest text-slate-400">
          Fault Frequency Hits
        </span>
        <div className="text-right font-mono text-xs text-slate-400">
          <div>
            dominant{' '}
            <span className="font-semibold text-warning">
              {detail.dominant_fault.toUpperCase()}
            </span>
          </div>
          <div className="text-[10px] text-slate-500">{detail.dominant_fault_hz.toFixed(2)} Hz</div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={130}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="name"
            tick={{ fontSize: 9, fill: '#64748b', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 9, fill: '#64748b', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={20}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="value" radius={[3, 3, 0, 0]} isAnimationActive>
            {chartData.map((entry) => (
              <Cell
                key={entry.key}
                fill={FAULT_COLORS[entry.key as FaultType]}
                fillOpacity={entry.key === detail.dominant_fault ? 1 : 0.45}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-4 gap-2">
        {chartData.map((entry) => (
          <div key={entry.key} className="text-center">
            <div
              className="font-mono text-[10px] font-semibold"
              style={{ color: FAULT_COLORS[entry.key as FaultType] }}
            >
              {entry.hz.toFixed(1)} Hz
            </div>
            <div className="font-mono text-[9px] text-slate-500">{entry.key.toUpperCase()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
