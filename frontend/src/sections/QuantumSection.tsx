import React, { useEffect, useRef, memo } from 'react';
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
const W = 700, H = 270;

// Static circuit layout
const CIRCUIT = [
  // Column 1: Hadamard (encoded as RX 90°)
  { type: 'gate', col: 120, qIdx: 0, label: 'H',  color: '#3E6AE1' },
  { type: 'gate', col: 120, qIdx: 1, label: 'H',  color: '#3E6AE1' },
  { type: 'gate', col: 120, qIdx: 2, label: 'H',  color: '#3E6AE1' },
  { type: 'gate', col: 120, qIdx: 3, label: 'H',  color: '#3E6AE1' },
  // Column 2: RX angle encoding
  { type: 'gate', col: 220, qIdx: 0, label: 'RX', color: '#059669' },
  { type: 'gate', col: 220, qIdx: 1, label: 'RY', color: '#059669' },
  { type: 'gate', col: 220, qIdx: 2, label: 'RX', color: '#059669' },
  { type: 'gate', col: 220, qIdx: 3, label: 'RY', color: '#059669' },
  // Column 3: CNOT entanglement
  { type: 'cnot', col: 330, ctrl: 0, tgt: 1 },
  { type: 'cnot', col: 330, ctrl: 2, tgt: 3 },
  // Column 4: variational layer
  { type: 'gate', col: 430, qIdx: 0, label: 'RZ', color: '#7C3AED' },
  { type: 'gate', col: 430, qIdx: 1, label: 'RZ', color: '#7C3AED' },
  { type: 'gate', col: 430, qIdx: 2, label: 'RX', color: '#7C3AED' },
  { type: 'gate', col: 430, qIdx: 3, label: 'RY', color: '#7C3AED' },
  // Column 5: second CNOT
  { type: 'cnot', col: 540, ctrl: 1, tgt: 2 },
  // Column 6: measurement
  { type: 'gate', col: 640, qIdx: 0, label: 'M', color: '#171A20' },
  { type: 'gate', col: 640, qIdx: 1, label: 'M', color: '#171A20' },
  { type: 'gate', col: 640, qIdx: 2, label: 'M', color: '#171A20' },
  { type: 'gate', col: 640, qIdx: 3, label: 'M', color: '#171A20' },
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

      {/* Circuit elements */}
      {CIRCUIT.map((el, i) => {
        if (el.type === 'gate') {
          const y = QUBITS[el.qIdx!];
          return (
            <motion.g
              key={i}
              initial={animated ? { opacity: 0, scale: 0.5 } : { opacity: 1, scale: 1 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.04, duration: 0.3 }}
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
              transition={{ delay: i * 0.04, duration: 0.3 }}
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

// ── Info cards ──────────────────────────────────────────────
const INFO = [
  { label: 'Architecture', value: '4-Qubit VQC', sub: 'Variational Quantum Circuit' },
  { label: 'Encoding',     value: 'Angle',       sub: 'RX/RY rotation on kinematics' },
  { label: 'Entanglement', value: 'CNOT × 2',    sub: 'q0↔q1  and  q1↔q2' },
  { label: 'Measurement',  value: '18 outputs',  sub: '3 trajectory waypoints × 6 features' },
];

export default function QuantumSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  const telemetry = useTelemetryStore(s => s.telemetry);
  const variance = telemetry?.metrics?.qnn?.variance?.toFixed(4) ?? '—';

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
          <h2 className="tesla-h2 mt-2">How the 4-Qubit VQC Works</h2>
          <p className="tesla-body mt-3">
            Kinematic features (velocity, position, acceleration) are angle-encoded into qubit rotations.
            CNOT gates entangle qubits, enabling the circuit to explore exponentially many trajectory
            hypotheses simultaneously.
          </p>
        </motion.div>

        {/* Circuit */}
        <div ref={ref}>
          <QuantumCircuitSVG animated={inView} />
        </div>

        {/* Live variance indicator */}
        <div className="mt-4 flex items-center gap-3">
          <span className="text-xs font-mono text-[#5C5E62]">Live shot variance:</span>
          <span className="text-xs font-mono font-bold text-[#6366F1]">{variance}</span>
          <span className="text-xs text-[#5C5E62]">(lower = higher confidence)</span>
        </div>

        {/* Info grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-10">
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
            { color: '#3E6AE1', label: 'Hadamard / Basis prep' },
            { color: '#059669', label: 'Angle encoding (kinematics)' },
            { color: '#7C3AED', label: 'Variational layer' },
            { color: '#6366F1', label: 'CNOT entanglement' },
            { color: '#171A20', label: 'Measurement' },
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
