import React, { useRef, useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { useTelemetryStore } from '../store/telemetryStore';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar as RechartsBar, PieChart, Pie, Cell,
} from 'recharts';

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

const LANE_COLORS = ['#6366F1', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#EC4899', '#14B8A6'];

export default function DatasetExplorerSection() {
  const telemetry = useTelemetryStore(s => s.telemetry);
  const frame = telemetry?.frame ?? '—';
  const speed = telemetry?.ego?.speed ?? '—';
  const objCount = telemetry?.objects?.length ?? 0;

  const [datasetData, setDatasetData] = useState<any>(null);
  const [edaData, setEdaData] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/data')
      .then(res => res.json())
      .then(data => setDatasetData(data))
      .catch(err => console.error('Failed to fetch dataset:', err));

    fetch('http://localhost:8000/api/eda')
      .then(res => res.json())
      .then(data => setEdaData(data))
      .catch(err => console.error('Failed to fetch EDA:', err));
  }, []);

  const totalRecords = datasetData?.total_records ?? 0;
  const totalFrames = datasetData?.total_frames ?? 0;
  const vehicleCount = datasetData?.vehicle_count ?? 0;
  const avgVel = datasetData?.avg_velocity ?? 0;
  const maxVel = datasetData?.max_velocity ?? 0;
  const minVel = datasetData?.min_velocity ?? 0;
  const stdVel = datasetData?.std_velocity ?? 0;
  const columnCount = datasetData?.column_count ?? 0;
  const sampleRows = datasetData?.sample_rows ?? [];
  const sampleCols = datasetData?.sample_cols ?? [];
  const scatter = datasetData?.scatter ?? [];

  const velDist = edaData?.velocity_distribution ?? [];
  const accDist = edaData?.acceleration_distribution ?? [];
  const laneDist = edaData?.lane_distribution ?? [];
  const headwayDist = edaData?.headway_distribution ?? [];
  const scenarioCounts = edaData?.scenario_row_counts ?? {};

  return (
    <section id="dataset" className="bg-[#FAFAFA] section-pad border-t border-[#E0E0E0]">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="mb-10 max-w-2xl"
        >
          <span className="tesla-label">Dataset Explorer</span>
          <h2 className="tesla-h2 mt-2">NGSIM US-101 Dataset</h2>
          <p className="tesla-body mt-3">
            Trained on the Next Generation Simulation (NGSIM) US-101 highway dataset
            containing {totalRecords.toLocaleString()} trajectory records from {vehicleCount.toLocaleString()} unique
            vehicles across {totalFrames.toLocaleString()} frames, sampled at 10 Hz with sub-meter precision.
          </p>
        </motion.div>

        {/* Live telemetry cards */}
        <div className="flex flex-wrap gap-4 mb-8">
          {[
            { l: 'Current Frame',    v: frame,     mono: true },
            { l: 'Ego Speed (ft/s)', v: typeof speed === 'number' ? speed.toFixed(1) : speed, mono: true },
            { l: 'Tracked Objects',  v: objCount,  mono: true },
            { l: 'WS Status',        v: telemetry ? 'Live' : 'No feed', mono: false },
          ].map(({ l, v, mono }) => (
            <div key={l} className="tesla-card flex-1 min-w-[140px]">
              <span className="tesla-label text-[10px]">{l}</span>
              <div className={`text-xl font-bold mt-1 ${mono ? 'font-mono' : ''} text-[#171A20]`}>{v}</div>
            </div>
          ))}
        </div>

        {/* Dataset statistics grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          {[
            { label: 'Total Records', value: totalRecords, sub: `${columnCount} features` },
            { label: 'Unique Vehicles', value: vehicleCount, sub: 'Tracked objects' },
            { label: 'Total Frames', value: totalFrames, sub: '10 Hz sampling' },
            { label: 'Avg Velocity', value: avgVel, sub: `σ = ${stdVel} ft/s`, suffix: ' ft/s' },
            { label: 'Max Velocity', value: maxVel, sub: 'Peak speed', suffix: ' ft/s' },
            { label: 'Min Velocity', value: minVel, sub: 'Lowest speed', suffix: ' ft/s' },
            { label: 'Avg Acceleration', value: datasetData?.avg_acceleration ?? 0, sub: `σ = ${datasetData?.std_acceleration ?? 0}`, suffix: '' },
            { label: 'Lane Count', value: datasetData?.lane_ids?.length ?? 0, sub: `IDs: ${datasetData?.lane_ids?.join(', ') ?? '—'}` },
          ].map((stat) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="tesla-card"
            >
              <div className="text-xl font-bold text-[#171A20] tracking-tight">
                <Count to={typeof stat.value === 'number' ? stat.value : 0} suffix={stat.suffix ?? ''} />
              </div>
              <div className="text-xs text-[#5C5E62] mt-1">{stat.label}</div>
              <div className="text-[10px] text-[#A1A1A1]">{stat.sub}</div>
            </motion.div>
          ))}
        </div>

        {/* EDA Charts: 2x2 grid */}
        <h3 className="text-sm font-semibold text-[#5C5E62] mb-4 uppercase tracking-widest">
          Exploratory Data Analysis
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* Velocity Distribution */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Velocity Distribution</h4>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <BarChart data={velDist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <RechartsBar dataKey="count" fill="#6366F1" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Acceleration Distribution */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Acceleration Distribution</h4>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <BarChart data={accDist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <RechartsBar dataKey="count" fill="#10B981" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Lane Distribution (Pie) */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Lane Distribution</h4>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <PieChart>
                  <Pie data={laneDist} dataKey="count" nameKey="lane" cx="50%" cy="50%"
                    outerRadius={70} label={({ lane, percent }) => `Lane ${lane} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {laneDist.map((_: any, i: number) => (
                      <Cell key={i} fill={LANE_COLORS[i % LANE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trajectory Scatter */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Trajectory Scatter (Local X vs Y)</h4>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <ScatterChart data={scatter}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                  <XAxis type="number" dataKey="x" tick={{ fontSize: 10 }} name="Local X" />
                  <YAxis type="number" dataKey="y" tick={{ fontSize: 10 }} name="Local Y" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter fill="#6366F1" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Headway + Scenario Counts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* Space Headway Distribution */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Space Headway Distribution</h4>
            <div className="h-[180px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <BarChart data={headwayDist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <RechartsBar dataKey="count" fill="#F59E0B" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Scenario Row Counts */}
          <div className="tesla-card">
            <h4 className="text-sm font-semibold text-[#171A20] mb-3">Scenario Data Availability</h4>
            <div className="h-[180px] w-full">
              <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                <BarChart data={Object.entries(scenarioCounts).map(([k, v]) => ({ scenario: k.replace('_', ' '), count: v }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
                  <XAxis dataKey="scenario" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <RechartsBar dataKey="count" fill="#8B5CF6" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Sample data table */}
        <div>
          <h3 className="text-sm font-semibold text-[#5C5E62] mb-4 uppercase tracking-widest">
            Sample Data Preview <span className="text-[#A1A1A1] font-normal">(First 10 rows from dataset)</span>
          </h3>
          <div className="tesla-card overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#E0E0E0]">
                  {sampleCols.map((col: string) => (
                    <th key={col} className="text-left text-xs font-semibold text-[#5C5E62] py-2 pr-6 whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sampleRows.map((row: any[], i: number) => (
                  <tr key={i} className="border-b border-[#F4F4F4] last:border-0 hover:bg-[#FAFAFA] transition-colors">
                    {row.map((cell: any, j: number) => (
                      <td key={j} className="py-2.5 pr-6 font-mono text-xs text-[#171A20] whitespace-nowrap">
                        {typeof cell === 'number' ? (Number.isInteger(cell) ? cell : cell.toFixed(2)) : cell}
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
