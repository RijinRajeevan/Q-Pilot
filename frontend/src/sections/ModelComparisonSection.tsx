import React, { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTelemetryStore } from '../store/telemetryStore';
import type { Scenario } from '../store/telemetryStore';

// ─── Animated number ──────────────────────────────────────
function Count({ to, decimals = 2, suffix = '' }: { to: number; decimals?: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    const st = performance.now();
    const dur = 1400;
    const prev = val;
    const tick = (now: number) => {
      const t = Math.min((now - st) / dur, 1);
      const e = 1 - (1 - t) ** 3;
      setVal(prev + (to - prev) * e);
      if (t < 1) animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [to]);

  return <span>{val.toFixed(decimals)}{suffix}</span>;
}

// ─── Animated metric bar ───────────────────────────────────
function Bar({ label, value, max, color, delay = 0, suffix = '' }: {
  label: string; value: number; max: number; color: string; delay?: number; suffix?: string;
}) {
  const pct = Math.min(Math.abs(value) / max, 1) * 100;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-end">
        <span className="text-sm font-medium text-[#171A20]">{label}</span>
        <span className="text-sm font-bold tabular-nums font-mono" style={{ color }}>
          <Count to={value} decimals={4} />{suffix}
        </span>
      </div>
      <div className="h-2.5 bg-[#F0F0F0] rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          whileInView={{ width: `${pct}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  );
}

// ─── Model config ──────────────────────────────────────────
interface ModelResult {
  name: string;
  r2: number;
  mse: number;
  ade: number;
  color: string;
  icon: string;
}

const MODEL_COLORS: Record<string, { color: string; icon: string }> = {
  'QNN (Qiskit VQC)':   { color: '#6366F1', icon: '⚛️' },
  'Random Forest':       { color: '#8B5CF6', icon: '🌲' },
  'Decision Tree':       { color: '#10B981', icon: '🌳' },
  'Linear Regression':   { color: '#9CA3AF', icon: '📈' },
};

export default function ModelComparisonSection() {
  const activeScenario = useTelemetryStore(s => s.activeScenario);
  const [apiData, setApiData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`http://localhost:8000/api/predict?scenario=${activeScenario}`)
      .then(res => res.json())
      .then(data => { setApiData(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [activeScenario]);

  // Extract from API response
  const qnn = apiData?.qnn ?? { r2: 0, mse: 0, ade: 0 };
  const rf  = apiData?.random_forest ?? { r2: 0, mse: 0, ade: 0 };
  const dt  = apiData?.decision_tree ?? { r2: 0, mse: 0, ade: 0 };
  const lr  = apiData?.linear_regression ?? { r2: 0, mse: 0, ade: 0 };
  const winner = apiData?.winner ?? '—';
  const improvement = apiData?.improvement_pct ?? 0;
  const sampleSize = apiData?.sample_size ?? 0;
  const trainSize = apiData?.train_size ?? 0;
  const testSize = apiData?.test_size ?? 0;
  const trainingTime = apiData?.training_time ?? 0;

  const models: ModelResult[] = [
    { name: 'QNN (Qiskit VQC)', r2: qnn.r2, mse: qnn.mse, ade: qnn.ade, ...MODEL_COLORS['QNN (Qiskit VQC)'] },
    { name: 'Random Forest',     r2: rf.r2,  mse: rf.mse,  ade: rf.ade,  ...MODEL_COLORS['Random Forest'] },
    { name: 'Decision Tree',     r2: dt.r2,  mse: dt.mse,  ade: dt.ade,  ...MODEL_COLORS['Decision Tree'] },
    { name: 'Linear Regression', r2: lr.r2,  mse: lr.mse,  ade: lr.ade,  ...MODEL_COLORS['Linear Regression'] },
  ];

  return (
    <section id="models" className="bg-white section-pad border-t border-[#E0E0E0]">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}
          className="mb-10 max-w-2xl"
        >
          <span className="tesla-label">Model Comparison</span>
          <h2 className="tesla-h2 mt-2">QNN vs Classical Models</h2>
          <p className="tesla-body mt-3">
            All metrics computed from real NGSIM trajectory data filtered per scenario.
            Models are trained on {sampleSize.toLocaleString()} data points with 80/20 train-test split.
          </p>
        </motion.div>

        {/* Training info banner */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeScenario}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
            className="mb-8 bg-[#EEF2FF] border border-[#C7D2FE] rounded-2xl px-6 py-5"
          >
            {loading ? (
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-[#6366F1] border-t-transparent rounded-full animate-spin" />
                <span className="text-sm text-[#6366F1] font-medium">Training models for {activeScenario.replace('_', ' ')}...</span>
              </div>
            ) : (
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-[#6366F1] font-semibold mb-1">
                    Best Model for <span className="capitalize">{activeScenario.replace('_', ' ')}</span>
                  </div>
                  <div className="text-2xl font-bold text-[#171A20] flex items-center gap-2">
                    <span>{MODEL_COLORS[winner]?.icon ?? '🏆'}</span> {winner}
                  </div>
                </div>
                <div className="flex gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#6366F1]">+{improvement}%</div>
                    <div className="text-[10px] uppercase text-[#5C5E62] font-semibold">vs Linear Reg.</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#171A20]">{trainSize.toLocaleString()}</div>
                    <div className="text-[10px] uppercase text-[#5C5E62] font-semibold">Train Samples</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#171A20]">{testSize.toLocaleString()}</div>
                    <div className="text-[10px] uppercase text-[#5C5E62] font-semibold">Test Samples</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#171A20]">{trainingTime}s</div>
                    <div className="text-[10px] uppercase text-[#5C5E62] font-semibold">Train Time</div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* R² + MSE + ADE side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* R² bars */}
          <AnimatePresence mode="wait">
            <motion.div
              key={`r2-${activeScenario}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-[#FAFAFA] border border-[#E0E0E0] rounded-2xl p-5"
            >
              <h3 className="text-sm font-semibold text-[#171A20] mb-5">
                R² Score <span className="text-[#5C5E62] text-xs font-normal">(higher = better)</span>
              </h3>
              <div className="flex flex-col gap-4">
                {models.map((m, i) => (
                  <Bar key={m.name} label={m.name} value={m.r2} max={1.0} color={m.color} delay={i * 0.08} />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* MSE bars */}
          <AnimatePresence mode="wait">
            <motion.div
              key={`mse-${activeScenario}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-[#FAFAFA] border border-[#E0E0E0] rounded-2xl p-5"
            >
              <h3 className="text-sm font-semibold text-[#171A20] mb-5">
                MSE <span className="text-[#5C5E62] text-xs font-normal">(lower = better)</span>
              </h3>
              <div className="flex flex-col gap-4">
                {models.map((m, i) => (
                  <Bar key={m.name} label={m.name} value={m.mse} max={Math.max(...models.map(x => x.mse), 0.01)} color={m.color} delay={i * 0.08} />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* ADE bars */}
          <AnimatePresence mode="wait">
            <motion.div
              key={`ade-${activeScenario}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-[#FAFAFA] border border-[#E0E0E0] rounded-2xl p-5"
            >
              <h3 className="text-sm font-semibold text-[#171A20] mb-5">
                ADE <span className="text-[#5C5E62] text-xs font-normal">(avg displacement error)</span>
              </h3>
              <div className="flex flex-col gap-4">
                {models.map((m, i) => (
                  <Bar key={m.name} label={m.name} value={m.ade} max={Math.max(...models.map(x => x.ade), 0.01)} color={m.color} delay={i * 0.08} />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Model detail cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {models.map((m, idx) => (
            <motion.div
              key={m.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.08 }}
              whileHover={{ y: -3 }}
              className={`bg-white rounded-2xl border p-5 transition-shadow hover:shadow-md relative
                ${winner === m.name ? 'border-[#6366F1]/50 bg-[#FAFAFF] ring-1 ring-[#6366F1]/20' : 'border-[#E0E0E0]'}`}
            >
              {winner === m.name && (
                <div className="absolute top-3 right-3 text-[9px] font-bold px-2.5 py-0.5 rounded-full text-white bg-[#6366F1]">
                  ★ Best
                </div>
              )}
              <div className="flex items-center gap-2 mb-4">
                <span className="text-xl">{m.icon}</span>
                <span className="text-sm font-semibold text-[#171A20]">{m.name}</span>
              </div>
              {[
                { l: 'R² Score', v: m.r2.toFixed(4) },
                { l: 'MSE', v: m.mse.toFixed(4) },
                { l: 'ADE', v: m.ade.toFixed(4) },
              ].map(({ l, v }) => (
                <div key={l} className="flex justify-between text-xs py-1.5 border-t border-[#F4F4F4]">
                  <span className="text-[#5C5E62]">{l}</span>
                  <span className="font-mono font-bold text-[#171A20]">{v}</span>
                </div>
              ))}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
