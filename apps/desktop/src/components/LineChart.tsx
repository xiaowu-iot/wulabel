/** 轻量折线图（SVG），用于训练曲线 */
interface Series {
  name: string;
  color: string;
  values: (number | null)[];
}

interface Props {
  series: Series[];
  /** 横轴总长度（比如总轮数），用来在训练中留出右侧空间 */
  xMax: number;
  yMin?: number;
  yMax?: number;
  height?: number;
  format?: (v: number) => string;
}

export default function LineChart({ series, xMax, yMin, yMax, height = 200, format = (v) => v.toFixed(2) }: Props) {
  const W = 600;
  const H = height;
  const pad = { l: 44, r: 12, t: 10, b: 22 };
  const all = series.flatMap((s) => s.values.filter((v): v is number => v != null && Number.isFinite(v)));
  const lo = yMin ?? (all.length ? Math.min(...all) : 0);
  let hi = yMax ?? (all.length ? Math.max(...all) : 1);
  if (hi - lo < 1e-9) hi = lo + 1;
  const n = Math.max(xMax, 1);
  const x = (i: number) => pad.l + ((i + 1) / n) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  const xTicks = Array.from(new Set([1, Math.ceil(n / 4), Math.ceil(n / 2), Math.ceil((3 * n) / 4), n])).filter((v) => v >= 1);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
          <text x={pad.l - 6} y={y(t) + 4} fill="var(--dim)" fontSize={11} textAnchor="end">
            {format(t)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text key={t} x={x(t - 1)} y={H - 6} fill="var(--dim)" fontSize={11} textAnchor="middle">
          {t}
        </text>
      ))}
      {series.map((s) => {
        const pts = s.values
          .map((v, i) => (v == null || !Number.isFinite(v) ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`))
          .filter(Boolean);
        return (
          <g key={s.name}>
            <polyline points={pts.join(" ")} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
            {pts.length === 1 && <circle cx={pts[0]!.split(",")[0]} cy={pts[0]!.split(",")[1]} r={3} fill={s.color} />}
          </g>
        );
      })}
    </svg>
  );
}
