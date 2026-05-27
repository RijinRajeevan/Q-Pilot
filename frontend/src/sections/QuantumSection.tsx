import React, { useRef, memo } from 'react';
import { motion, useInView } from 'framer-motion';
import { useTelemetryStore } from '../store/telemetryStore';

// ── Gate SVG helpers ────────────────────────────────────────
const GATE_W = 32, GATE_H = 26;

function Gate({ x, y, label, color }: { x: number; y: number; label: string; color: string }) {
  return (
    <g>
      <rect
        x={x - GATE_W / 2} y={y - GATE_H / 2}
        width={GATE_W} height={GATE_H}
        rx={4} fill={color} stroke="none"
      />
      <text x={x} y={y + 5} textAnchor="middle"
        fontSize={10} fontWeight="700" fill="#fff" fontFamily="Inter, system-ui">
        {label}
      </text>
    </g>
  );
}

function CNOT({ x, ctrl, tgt }: { x: number; ctrl: number; tgt: number }) {
  return (
    <g>
      <line x1={x} y1={ctrl} x2={x} y2={tgt} stroke="#6366F1" strokeWidth={2} strokeDasharray="4 2" />
      <circle cx={x} cy={ctrl} r={5} fill="#6366F1" />
      <circle cx={x} cy={tgt} r={10} fill="none" stroke="#6366F1" strokeWidth={2} />
      <line x1={x - 10} y1={tgt} x2={x + 10} y2={tgt} stroke="#6366F1" strokeWidth={2} />
      <line x1={x} y1={tgt - 10} x2={x} y2={tgt + 10} stroke="#6366F1" strokeWidth={2} />
    </g>
  );
}

// 4 qubits, rows at y=60,110,160,210
const QUBITS = [60, 110, 160, 210];
const W = 780, H = 270;

// V7 Circuit layout — matches the real qnn_regressor.py architecture:
// RY angle encoding → Variational Layer 1 (RY+RZ) → CNOT ring → Var Layer 2 (RY+RZ) → CNOT ring → Measurement
const CIRCUIT = [
  // Column 1: RY angle encoding (input features)
  { type: 'gate', col: 80,  qIdx: 0, label: 'RY', color: '#059669' },
  { type: 'gate', col: 80,  qIdx: 1, label: 'RY', color: '#059669' },
  { type: 'gate', col: 80,  qIdx: 2, label: 'RY', color: '#059669' },
  { type: 'gate', col: 80,  qIdx: 3, label: 'RY', color: '#059669' },
  // Column 2: Variational Layer 1 (RY + RZ rotations)
  { type: 'gate', col: 170, qIdx: 0, label: 'RY', color: '#7C3AED' },
  { type: 'gate', col: 170, qIdx: 1, label: 'RY', color: '#7C3AED' },
  { type: 'gate', col: 170, qIdx: 2, label: 'RY', color: '#7C3AED' },
  { type: 'gate', col: 170, qIdx: 3, label: 'RY', color: '#7C3AED' },
  { type: 'gate', col: 230, qIdx: 0, label: 'RZ', color: '#7C3AED' },
  { type: 'gate', col: 230, qIdx: 1, label: 'RZ', color: '#7C3AED' },
  { type: 'gate', col: 230, qIdx: 2, label: 'RZ', color: '#7C3AED' },
  { type: 'gate', col: 230, qIdx: 3, label: 'RZ', color: '#7C3AED' },
  // Column 3: CNOT ring entanglement (q0→q1, q1→q2, q2→q3, q3→q0)
  { type: 'cnot', col: 310, ctrl: 0, tgt: 1 },
  { type: 'cnot', col: 360, ctrl: 2, tgt: 3 },
  { type: 'cnot', col: 340, ctrl: 1, tgt: 2 },
  // Column 4: Variational Layer 2 (RY + RZ)
  { type: 'gate', col: 430, qIdx: 0, label: 'RY', color: '#DC2626' },
  { type: 'gate', col: 430, qIdx: 1, label: 'RY', color: '#DC2626' },
  { type: 'gate', col: 430, qIdx: 2, label: 'RY', color: '#DC2626' },
  { type: 'gate', col: 430, qIdx: 3, label: 'RY', color: '#DC2626' },
  { type: 'gate', col: 490, qIdx: 0, label: 'RZ', color: '#DC2626' },
  { type: 'gate', col: 490, qIdx: 1, label: 'RZ', color: '#DC2626' },
  { type: 'gate', col: 490, qIdx: 2, label: 'RZ', color: '#DC2626' },
  { type: 'gate', col: 490, qIdx: 3, label: 'RZ', color: '#DC2626' },
  // Column 5: Second CNOT ring
  { type: 'cnot', col: 570, ctrl: 0, tgt: 1 },
  { type: 'cnot', col: 620, ctrl: 2, tgt: 3 },
  { type: 'cnot', col: 600, ctrl: 1, tgt: 2 },
  // Column 6: Measurement (Z expectation values)
  { type: 'gate', col: 720, qIdx: 0, label: '⟨Z⟩', color: '#171A20' },
  { type: 'gate', col: 720, qIdx: 1, label: '⟨Z⟩', color: '#171A20' },
  { type: 'gate', col: 720, qIdx: 2, label: '⟨Z⟩', color: '#171A20' },
  { type: 'gate', col: 720, qIdx: 3, label: '⟨Z⟩', color: '#171A20' },
];

function QuantumCircuitSVG({ animated }: { animated: boolean }) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full rounded-xl border border-[#E0E0E0] bg-white overflow-hidden"
      style={{ maxHeight: 280 }}
    >
      {/* Qubit wires */}
      {QUBITS.map((y, i) => (
        <g key={i}>
          <line x1={30} y1={y} x2={W - 30} y2={y} stroke="#E0E0E0" strokeWidth={1.5} />
          <text x={22} y={y + 5} textAnchor="middle" fontSize={11} fill="#5C5E62" fontFamily="Inter" fontWeight="600">
            q{i}
          </text>
        </g>
      ))}

      {/* Column labels */}
      <text x={80}  y={30} textAnchor="middle" fontSize={9} fill="#5C5E62" fontFamily="Inter" fontWeight="600">INPUT</text>
      <text x={200} y={30} textAnchor="middle" fontSize={9} fill="#7C3AED" fontFamily="Inter" fontWeight="600">VAR LAYER 1</text>
      <text x={335} y={30} textAnchor="middle" fontSize={9} fill="#6366F1" fontFamily="Inter" fontWeight="600">ENTANGLE</text>
      <text x={460} y={30} textAnchor="middle" fontSize={9} fill="#DC2626" fontFamily="Inter" fontWeight="600">VAR LAYER 2</text>
      <text x={595} y={30} textAnchor="middle" fontSize={9} fill="#6366F1" fontFamily="Inter" fontWeight="600">ENTANGLE</text>
      <text x={720} y={30} textAnchor="middle" fontSize={9} fill="#171A20" fontFamily="Inter" fontWeight="600">MEASURE</text>

      {/* Circuit elements */}
      {CIRCUIT.map((el, i) => {
        if (el.type === 'gate') {
          const y = QUBITS[el.qIdx!];
          return (
            <motion.g
              key={i}
              initial={animated ? { opacity: 0, scale: 0.5 } : { opacity: 1, scale: 1 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.03, duration: 0.3 }}
            >
              <Gate x={el.col!} y={y} label={el.label!} color={el.color!} />
            </motion.g>
          );
        }
        if (el.type === 'cnot') {
          const cy = QUBITS[el.ctrl!];
          const ty = QUBITS[el.tgt!];
          return (
            <motion.g
              key={i}
              initial={animated ? { opacity: 0 } : { opacity: 1 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.03, duration: 0.3 }}
            >
              <CNOT x={el.col!} ctrl={cy} tgt={ty} />
            </motion.g>
          );
        }
        return null;
      })}
    </svg>
  );
}

// ── Info cards — matches real qnn_regressor.py ──────────────
const INFO = [
  { label: 'Architecture', value: '4-Qubit VQC', sub: 'Variational Quantum Circuit' },
  { label: 'Encoding',     value: 'RY Angle',    sub: 'Feature → qubit rotation angle' },
  { label: 'Entanglement', value: 'CNOT Ring',   sub: 'q0→q1→q2→q3→q0 (circular)' },
  { label: 'Parameters',   value: '16 weights',  sub: '2 layers × 4 qubits × 2 (RY+RZ)' },
  { label: 'Optimizer',    value: 'COBYLA',      sub: 'Constrained optimization' },
  { label: 'Measurement',  value: '⟨Z⟩ × 4',     sub: 'Expectation values on all qubits' },
];

export default function QuantumSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  const telemetry = useTelemetryStore(s => s.telemetry);
  const activeScenario = useTelemetryStore(s => s.activeScenario);
  const variance = telemetry?.metrics?.qnn?.variance?.toFixed(4) ?? '—';
  const qnnLatency = telemetry?.metrics?.qnn?.latency?.toFixed(0) ?? '—';
  const qnnAde = telemetry?.metrics?.qnn?.ade?.toFixed(3) ?? '—';

  return (
    <section id="quantum" className="bg-[#FAFAFA] section-pad border-t border-[#E0E0E0]">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="mb-12 max-w-2xl"
        >
          <span className="tesla-label">Quantum Circuit</span>
          <h2 className="tesla-h2 mt-2">Real 4-Qubit Variational Quantum Circuit</h2>
          <p className="tesla-body mt-3">
            Built with Qiskit — 4 input features are angle-encoded via RY gates, then processed through
            2 variational layers (RY+RZ rotations) with CNOT ring entanglement. The circuit uses 16 trainable
            parameters optimized with COBYLA on NGSIM trajectory data.
          </p>
        </motion.div>

        {/* Circuit */}
        <div ref={ref}>
          <QuantumCircuitSVG animated={inView} />
        </div>

        {/* Live metrics panel */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="tesla-card">
            <span className="tesla-label text-[10px]">Shot Variance</span>
            <div className="text-lg font-bold text-[#6366F1] mt-1 font-mono">{variance}</div>
            <div className="text-[10px] text-[#5C5E62]">lower = higher confidence</div>
          </div>
          <div className="tesla-card">
            <span className="tesla-label text-[10px]">QNN Latency</span>
            <div className="text-lg font-bold text-[#171A20] mt-1 font-mono">{qnnLatency}ms</div>
            <div className="text-[10px] text-[#5C5E62]">per-frame inference</div>
          </div>
          <div className="tesla-card">
            <span className="tesla-label text-[10px]">QNN ADE</span>
            <div className="text-lg font-bold text-[#171A20] mt-1 font-mono">{qnnAde}</div>
            <div className="text-[10px] text-[#5C5E62]">avg displacement error</div>
          </div>
          <div className="tesla-card">
            <span className="tesla-label text-[10px]">Scenario</span>
            <div className="text-lg font-bold text-[#171A20] mt-1 capitalize">{activeScenario.replace('_', ' ')}</div>
            <div className="text-[10px] text-[#5C5E62]">active prediction mode</div>
          </div>
        </div>

        {/* Info grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-10">
          {INFO.map(({ label, value, sub }) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="tesla-card"
            >
              <span className="tesla-label text-[10px]">{label}</span>
              <div className="text-lg font-bold text-[#171A20] mt-1">{value}</div>
              <div className="text-[11px] text-[#5C5E62] mt-0.5 leading-snug">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-5 mt-8">
          {[
            { color: '#059669', label: 'RY angle encoding (input features)' },
            { color: '#7C3AED', label: 'Variational Layer 1 (RY+RZ)' },
            { color: '#DC2626', label: 'Variational Layer 2 (RY+RZ)' },
            { color: '#6366F1', label: 'CNOT ring entanglement' },
            { color: '#171A20', label: '⟨Z⟩ expectation measurement' },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ background: color }} />
              <span className="text-xs text-[#5C5E62]">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
