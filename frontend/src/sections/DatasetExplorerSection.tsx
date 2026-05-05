import React, { useRef, useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { useTelemetryStore } from '../store/telemetryStore';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// ── Animated number ─────────────────────────────────────────
function Count({ to, suffix = '' }: { to: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  useEffect(() => {
    if (!inView) return;
    const st = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - st) / 1600, 1);
      const e = 1 - (1 - t) ** 3;
      setVal(Math.round(to * e));
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [inView, to]);
  return <span ref={ref}>{val.toLocaleString()}{suffix}</span>;
}

// ── Generate synthetic scatter from live telemetry ──────────
function buildScatterPoints(telemetry: any): { x: number; y: number }[] {
  if (!telemetry?.objects?.length) {
    return Array.from({ length: 60 }, (_, i) => ({
      x: Math.sin(i * 0.4) * 100 + i * 3,
      y: Math.cos(i * 0.3) * 80 + i * 2,
    }));
  }
  return telemetry.objects.flatMap((obj: any) =>
    (obj.final ?? []).map((pt: any) => ({ x: pt.x, y: pt.y }))
  );
}

const FEATURE_COLS = [
  'Frame ID', 'Vehicle ID', 'Local X', 'Local Y', 'Velocity (ft/s)', 'Acceleration',
];
const SAMPLE_ROWS = [
  [1,  1001, 52.1, 310.2, 22.4, -0.3],
  [1,  1002, 95.7, 285.9, 18.1,  0.1],
  [2,  1001, 52.4, 309.8, 22.2, -0.1],
  [2,  1003, 44.2, 195.3, 30.0,  1.2],
  [3,  1002, 96.0, 285.5, 17.9, -0.2],
];

export default function DatasetExplorerSection() {
  const telemetry = useTelemetryStore(s => s.telemetry);
  const points = buildScatterPoints(telemetry);
  const frame = telemetry?.frame ?? '—';
  const speed = telemetry?.ego?.speed ?? '—';
  const objCount = telemetry?.objects?.length ?? 0;

  const [datasetData, setDatasetData] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/data')
      .then(res => res.json())
      .then(data => setDatasetData(data))
      .catch(err => console.error('Failed to fetch dataset:', err));
  }, []);

  const totalFrames = datasetData?.total_frames ?? 1741;
  const vehicleCount = datasetData?.vehicle_count ?? 225;
  const avgVel = datasetData?.avg_velocity ?? 22.4;
  const maxVel = datasetData?.max_velocity ?? 55.1;

  const datasetStats = [
    { label: 'Total Frames', value: totalFrames.toString(), sub: 'In recording' },
    { label: 'Unique Vehicles', value: vehicleCount.toString(), sub: 'Tracked objects' },
    { label: 'Avg Velocity', value: `${avgVel} km/h`, sub: 'Traffic flow' },
    { label: 'Max Velocity', value: `${maxVel} km/h`, sub: 'Peak speed' },
  ];

  return (
    <section id="dataset" className="bg-[#FAFAFA] section-pad border-t border-[#E0E0E0]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="mb-12 max-w-2xl"
        >
          <span className="tesla-label">Dataset Explorer</span>
          <h2 className="tesla-h2 mt-2">NGSIM US-101 Dataset</h2>
          <p className="tesla-body mt-3">
            The model is trained on the Next Generation Simulation (NGSIM) US-101 highway
            dataset, containing detailed vehicle trajectories sampled at 10 Hz with
            sub-meter precision from multiple synchronized cameras.
          </p>
        </motion.div>

        <div className="flex flex-wrap gap-4 mb-10">
          {[
            { l: 'Current Frame',    v: frame,     mono: true },
            { l: 'Ego Speed (km/h)', v: typeof speed === 'number' ? speed.toFixed(1) : speed, mono: true },
            { l: 'Tracked Objects',  v: objCount,  mono: true },
            { l: 'WS Status',        v: telemetry ? 'Live' : 'No feed', mono: false },
          ].map(({ l, v, mono }) => (
            <div key={l} className="tesla-card flex-1 min-w-[140px]">
              <span className="tesla-label text-[10px]">{l}</span>
              <div className={`text-xl font-bold mt-1 ${mono ? 'font-mono' : ''} text-[#171A20]`}>{v}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div>
            <h3 className="text-sm font-semibold text-[#5C5E62] mb-3 uppercase tracking-widest">
              Live Trajectory Scatter
            </h3>
            <div className="h-[200px] w-full border border-[#E0E0E0] rounded-xl bg-white p-2">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <ScatterChart data={datasetData?.scatter || points}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" hide />
                  <YAxis type="number" dataKey="y" hide />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter fill="#6366F1" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-[#5C5E62] mb-3 uppercase tracking-widest">
              Dataset Statistics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {datasetStats.map((stat) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  className="tesla-card"
                >
                  <div className="flex items-end gap-3 mb-1">
                    <h3 className="text-xl font-bold text-[#171A20] tracking-tight">{stat.value}</h3>
                  </div>
                  <div className="text-xs text-[#5C5E62]">{stat.label}</div>
                  <div className="text-[10px] text-[#A1A1A1]">{stat.sub}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Sample data table */}
        <div>
          <h3 className="text-sm font-semibold text-[#5C5E62] mb-4 uppercase tracking-widest">
            Sample Data Preview
          </h3>
          <div className="tesla-card overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#E0E0E0]">
                  {FEATURE_COLS.map(col => (
                    <th key={col} className="text-left text-xs font-semibold text-[#5C5E62] py-2 pr-6 whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SAMPLE_ROWS.map((row, i) => (
                  <tr key={i} className="border-b border-[#F4F4F4] last:border-0 hover:bg-[#FAFAFA] transition-colors">
                    {row.map((cell, j) => (
                      <td key={j} className="py-2.5 pr-6 font-mono text-xs text-[#171A20] whitespace-nowrap">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
