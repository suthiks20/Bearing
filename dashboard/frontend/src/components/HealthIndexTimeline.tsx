import {
  Area,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TooltipProps = { active?: boolean; payload?: any[] };

export interface HIDataPoint {
  idx: number;
  hi_proxy: number;
}

interface HealthIndexTimelineProps {
  data: HIDataPoint[];
  currentIdx: number;
  onsetThreshold: number;
  failThreshold: number;
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as HIDataPoint;
  const hi = d.hi_proxy;
  const color = hi < 0.5 ? '#ef4444' : hi < 0.7 ? '#f59e0b' : '#22d3ee';
  return (
    <div className="glass-card px-3 py-2 text-xs font-mono space-y-0.5">
      <div className="text-slate-400">file {d.idx}</div>
      <div style={{ color }}>HI {hi.toFixed(4)}</div>
    </div>
  );
}

export default function HealthIndexTimeline({
  data,
  currentIdx,
  onsetThreshold,
  failThreshold,
}: HealthIndexTimelineProps) {
  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      <span className="font-mono text-xs uppercase tracking-widest text-slate-400">
        Health Index (proxy)
      </span>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="hiGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.25} />
              <stop offset={`${(1 - onsetThreshold) * 100}%`} stopColor="#f59e0b" stopOpacity={0.2} />
              <stop offset={`${(1 - failThreshold) * 100}%`} stopColor="#ef4444" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="idx"
            tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            minTickGap={80}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip content={<CustomTooltip />} />

          <Area
            type="monotone"
            dataKey="hi_proxy"
            stroke="none"
            fill="url(#hiGrad)"
            fillOpacity={1}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="hi_proxy"
            stroke="#22d3ee"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />

          <ReferenceLine
            y={onsetThreshold}
            stroke="#f59e0b"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{ value: 'onset', position: 'right', fontSize: 9, fill: '#f59e0b', fontFamily: 'monospace' }}
          />
          <ReferenceLine
            y={failThreshold}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{ value: 'fail', position: 'right', fontSize: 9, fill: '#ef4444', fontFamily: 'monospace' }}
          />
          <ReferenceLine
            x={currentIdx}
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
