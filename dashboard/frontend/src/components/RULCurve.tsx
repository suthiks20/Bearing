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

export interface RULDataPoint {
  idx: number;
  rul_estimate: number;
  rul_lower: number;
  rul_upper: number;
  band_width: number; // rul_upper - rul_lower, used for recharts stacked Area band
}

interface RULCurveProps {
  data: RULDataPoint[];
  currentIdx: number;
  maxLife: number;
}

function CustomTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as RULDataPoint;
  return (
    <div className="glass-card px-3 py-2 text-xs font-mono space-y-0.5">
      <div className="text-slate-400">file {d.idx}</div>
      <div className="text-healthy">RUL {d.rul_estimate.toFixed(1)}</div>
      <div className="text-slate-500">
        ±σ [{d.rul_lower.toFixed(0)}, {d.rul_upper.toFixed(0)}]
      </div>
    </div>
  );
}

export default function RULCurve({ data, currentIdx, maxLife }: RULCurveProps) {
  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      <span className="font-mono text-xs uppercase tracking-widest text-slate-400">
        RUL Estimate
      </span>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="rulBand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
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
            domain={[0, maxLife]}
            tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Uncertainty band via stacked Areas: floor (invisible) + width (visible) */}
          <Area
            type="monotone"
            dataKey="rul_lower"
            stackId="band"
            stroke="none"
            fill="transparent"
            fillOpacity={0}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="band_width"
            stackId="band"
            stroke="none"
            fill="url(#rulBand)"
            fillOpacity={1}
            isAnimationActive={false}
          />

          {/* Main RUL curve */}
          <Line
            type="monotone"
            dataKey="rul_estimate"
            stroke="#22d3ee"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />

          {/* Playhead */}
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
