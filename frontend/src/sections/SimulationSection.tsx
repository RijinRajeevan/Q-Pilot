import React, { useRef, useEffect, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTelemetryStore } from '../store/telemetryStore';
import type { Scenario, TrackedObject } from '../store/telemetryStore';

// ─── Scenario config ──────────────────────────────────────
export const SCENARIOS: { id: Scenario; label: string; icon: string; desc: string }[] = [
  { id: 'highway',        label: 'Highway',        icon: '🛣️',  desc: 'Open road, high-speed cruise' },
  { id: 'lane_change',    label: 'Lane Change',    icon: '↔️',  desc: 'Lateral maneuver detection' },
  { id: 'urban',          label: 'Urban Traffic',  icon: '🏙️',  desc: 'Dense scene, pedestrians' },
  { id: 'emergency_brake',label: 'Emergency Brake',icon: '🛑',  desc: 'Sudden deceleration event' },
  { id: 'sharp_turn',     label: 'Sharp Turn',     icon: '↩️',  desc: 'High-curvature prediction' },
];

// ─── Canvas overlay ───────────────────────────────────────
const OBJECT_COLORS: Record<string, string> = {
  car:       '#3B82F6',  // blue
  pedestrian:'#8B5CF6',  // violet
  truck:     '#F59E0B',  // amber
  cyclist:   '#10B981',  // green
};
const RISK_COLORS  = { safe:'#059669', caution:'#D97706', danger:'#DC2626' };

function drawOverlay(ctx: CanvasRenderingContext2D, objects: TrackedObject[]) {
  if (!objects?.length) return;

  objects.forEach((obj) => {
    const { x, y } = obj.current;
    const riskColor = RISK_COLORS[obj.risk ?? 'safe'];
    const objColor  = OBJECT_COLORS[obj.object_type ?? 'car'] ?? '#3B82F6';
    const hw = 44, hh = 28, cs = 14;

    // Corner-style bounding box (Waymo style)
    ctx.strokeStyle = objColor;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = objColor;
    ctx.shadowBlur = 6;
    [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx, sy]) => {
      const cx = x + sx * hw, cy = y + sy * hh;
      ctx.beginPath();
      ctx.moveTo(cx - sx * cs, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy - sy * cs);
      ctx.stroke();
    });
    ctx.shadowBlur = 0;

    // Risk dot
    ctx.beginPath();
    ctx.arc(x + hw - 6, y - hh + 6, 5, 0, Math.PI * 2);
    ctx.fillStyle = riskColor;
    ctx.fill();

    // Label chip: object type + behavior
    const typeLabel = (obj.object_type ?? 'car').toUpperCase();
    const behavior  = obj.behavior ?? '';
    const mainTxt   = typeLabel;
    ctx.font = 'bold 10px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(mainTxt).width;
    const chipX = x - hw;
    const chipY = y - hh - 24;
    ctx.fillStyle = objColor;
    ctx.beginPath();
    ctx.roundRect(chipX, chipY, tw + 14, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(mainTxt, chipX + 7, chipY + 13);

    // Behavior sub-label
    if (behavior) {
      ctx.font = '9px Inter, system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      const bw = ctx.measureText(behavior).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.beginPath();
      ctx.roundRect(chipX, chipY + 21, bw + 12, 14, 3);
      ctx.fill();
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(behavior, chipX + 6, chipY + 32);
    }

    // QNN uncertainty cone (violet semi-transparent)
    if (obj.qnn?.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(99,102,241,0.55)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.moveTo(x, y);
      obj.qnn.forEach((pt) => ctx.lineTo(pt.x, pt.y));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.fillStyle = 'rgba(99,102,241,0.07)';
      ctx.moveTo(x, y);
      obj.qnn.forEach((pt) => ctx.lineTo(pt.x + (pt.uncert ?? 0), pt.y));
      for (let i = obj.qnn.length - 1; i >= 0; i--)
        ctx.lineTo(obj.qnn[i].x - (obj.qnn[i].uncert ?? 0), obj.qnn[i].y);
      ctx.closePath(); ctx.fill();
    }

    // Fusion trajectory path
    if (obj.final?.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = riskColor;
      ctx.lineWidth = 2;
      ctx.shadowColor = riskColor;
      ctx.shadowBlur = 6;
      ctx.moveTo(x, y);
      obj.final.forEach((pt) => ctx.lineTo(pt.x, pt.y));
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  });
}

// ─── Scenario pill ────────────────────────────────────────
const ScenarioPill = memo(function ScenarioPill({
  s, active, onClick,
}: { s: typeof SCENARIOS[0]; active: boolean; onClick: () => void }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      className={`relative flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 border
        ${active
          ? 'bg-[#171A20] text-white border-[#171A20] shadow-md'
          : 'bg-white text-[#5C5E62] border-[#E0E0E0] hover:border-[#171A20] hover:text-[#171A20]'
        }`}
    >
      <span>{s.icon}</span>
      <span>{s.label}</span>
      {active && (
        <motion.span
          layoutId="scenario-indicator"
          className="absolute inset-0 rounded-full bg-[#171A20] -z-10"
        />
      )}
    </motion.button>
  );
});

// ─── Risk explanation panel ────────────────────────────────
function RiskPanel({ objects, wsStatus }: { objects: TrackedObject[]; wsStatus: string }) {
  const mostCritical = objects.find(o => o.risk === 'danger')
    ?? objects.find(o => o.risk === 'caution')
    ?? objects[0];

  const isLive = wsStatus === 'connected';

  if (!isLive) {
    return (
      <div className="tesla-card flex flex-col items-center justify-center gap-3 min-h-[200px] text-center">
        <div className="w-10 h-10 rounded-full bg-[#F4F4F4] flex items-center justify-center text-xl">📡</div>
        <p className="text-sm text-[#5C5E62]">Simulation Offline</p>
        <p className="text-xs text-[#5C5E62]">Start the FastAPI backend to connect</p>
      </div>
    );
  }

  if (!mostCritical) {
    return (
      <div className="tesla-card flex flex-col items-center justify-center gap-3 min-h-[200px] text-center">
        <div className="w-10 h-10 rounded-full bg-emerald-50 flex items-center justify-center">
          <span className="text-emerald-600 text-xl">✓</span>
        </div>
        <p className="text-sm font-semibold text-emerald-700">Scene Clear</p>
        <p className="text-xs text-[#5C5E62]">No tracked objects detected yet</p>
      </div>
    );
  }

  const riskConfig = {
    safe:    { bg: 'bg-emerald-50',  text: 'text-emerald-700',  border: 'border-emerald-200', label: 'Safe',   dot: 'bg-emerald-500' },
    caution: { bg: 'bg-amber-50',    text: 'text-amber-700',    border: 'border-amber-200',   label: 'Caution', dot: 'bg-amber-400'  },
    danger:  { bg: 'bg-red-50',      text: 'text-red-700',      border: 'border-red-200',     label: 'Danger',  dot: 'bg-red-500'    },
  };
  const rc = riskConfig[mostCritical.risk];
  const objTypeIcon = { car:'🚗', pedestrian:'🚶', truck:'🚛', cyclist:'🚲' };

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={mostCritical.id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25 }}
        className="tesla-card flex flex-col gap-4"
      >
        {/* Risk badge */}
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${rc.bg} ${rc.border}`}>
          <span className={`w-2.5 h-2.5 rounded-full ${rc.dot}`} />
          <span className={`text-sm font-bold ${rc.text}`}>{rc.label}</span>
          <span className="ml-auto text-lg">{objTypeIcon[mostCritical.object_type ?? 'car']}</span>
        </div>

        {/* Object type */}
        <div>
          <div className="tesla-label text-[10px] mb-1">Detected Object</div>
          <div className="text-sm font-semibold text-[#171A20] capitalize">
            {(mostCritical.object_type ?? 'car')} #{mostCritical.id}
          </div>
          <div className="text-xs text-[#5C5E62] mt-0.5 capitalize">{mostCritical.behavior ?? 'Moving'}</div>
        </div>

        {/* Reason */}
        {mostCritical.risk_reason && (
          <div>
            <div className="tesla-label text-[10px] mb-1">Reason</div>
            <p className="text-sm text-[#171A20] leading-snug">{mostCritical.risk_reason}</p>
          </div>
        )}

        {/* Suggested action */}
        {mostCritical.suggested_action && (
          <div className="bg-[#F4F4F4] rounded-lg px-3 py-2">
            <div className="tesla-label text-[10px] mb-1 text-[#3E6AE1]">Suggested Action</div>
            <p className="text-sm font-medium text-[#171A20]">{mostCritical.suggested_action}</p>
          </div>
        )}

        {/* Confidence */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="tesla-label text-[10px]">Q-Confidence</span>
            <span className="font-mono font-bold text-[#171A20]">{(mostCritical.confidence ?? 80).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-[#E0E0E0] rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-[#3E6AE1] rounded-full"
              animate={{ width: `${mostCritical.confidence ?? 80}%` }}
              transition={{ type: 'spring', bounce: 0, duration: 0.6 }}
            />
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// ─── Metric card ──────────────────────────────────────────
const MetricCard = memo(function MetricCard({
  label, value, sub,
}: { label: string; value: string; sub?: string }) {
  return (
    <div className="tesla-card">
      <div className="tesla-label text-[10px]">{label}</div>
      <div className="text-xl font-bold text-[#171A20] mt-1 tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-[#5C5E62] font-mono mt-0.5">{sub}</div>}
    </div>
  );
});

// ─── Main component ───────────────────────────────────────
export default function SimulationSection({ sectionRef }: { sectionRef: React.RefObject<HTMLElement> }) {
  const telemetry = useTelemetryStore(s => s.telemetry);
  const wsStatus = useTelemetryStore(s => s.wsStatus);
  const activeScenario = useTelemetryStore(s => s.activeScenario);
  const setScenario = useTelemetryStore(s => s.setScenario);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const rafRef     = useRef<number | null>(null);
  const telRef     = useRef(telemetry);
  useEffect(() => { telRef.current = telemetry; }, [telemetry]);

  // rAF loop — only canvas overlay, video plays natively
  const renderLoop = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const t = telRef.current;
      if (t?.objects?.length) drawOverlay(ctx, t.objects);
    }
    rafRef.current = requestAnimationFrame(renderLoop);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(renderLoop);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [renderLoop]);

  const t       = telemetry;
  const isLive  = wsStatus === 'connected';
  const fps     = t?.fps?.toFixed(0) ?? '—';
  const latency = t?.system_latency?.toFixed(0) ?? '—';
  const ade     = t?.metrics?.qnn?.ade?.toFixed(3) ?? '—';
  const conf    = t?.metrics?.qnn?.variance != null
    ? `${Math.max(0, 100 - t.metrics.qnn.variance * 100).toFixed(0)}%` : '—';
  const objects = t?.objects ?? [];

  // sklearn live metrics from WebSocket
  const sk = t?.sklearn_metrics;
  const activeModel = sk?.winner ?? '—';
  const modelR2 = sk?.qnn_r2?.toFixed(4) ?? '—';
  const modelMSE = sk?.qnn_mse?.toFixed(3) ?? '—';

  // Scene-level risk
  const topRisk = objects.find(o => o.risk === 'danger') ? 'danger'
    : objects.find(o => o.risk === 'caution') ? 'caution' : 'safe';
  const riskLabel = { safe: 'All Clear', caution: 'Caution', danger: 'DANGER' }[topRisk];
  const riskCls   = { safe: 'risk-safe', caution: 'risk-caution', danger: 'risk-danger' }[topRisk];

  const activeSc = SCENARIOS.find(s => s.id === activeScenario)!;

  return (
    <section ref={sectionRef as React.RefObject<HTMLElement>} id="simulation" className="bg-white section-pad border-t border-[#E0E0E0]">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <span className="tesla-label">Live Simulation</span>
          <h2 className="tesla-h2 mt-2">Real-Time Intelligence</h2>
          <p className="tesla-body mt-2 max-w-lg">
            Quantum-enhanced trajectory prediction across 5 driving scenarios.
            Switch scenarios to see how the QNN adapts in real time.
          </p>
        </motion.div>

        {/* Scenario selector */}
        <div className="flex flex-wrap gap-2 mb-6">
          {SCENARIOS.map(s => (
            <ScenarioPill
              key={s.id} s={s}
              active={activeScenario === s.id}
              onClick={() => setScenario(s.id)}
            />
          ))}
        </div>

        {/* Active scenario info */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeScenario}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 12 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-3 mb-5"
          >
            <span className="text-xl">{activeSc.icon}</span>
            <div>
              <span className="text-sm font-semibold text-[#171A20]">{activeSc.label}</span>
              <span className="text-xs text-[#5C5E62] ml-2">— {activeSc.desc}</span>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400 animate-pulse'}`} />
              <span className="text-xs font-mono text-[#5C5E62]">
                {isLive ? `Live · ${fps} FPS` : wsStatus === 'error' ? 'Offline' : 'Connecting…'}
              </span>
              <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full ${riskCls}`}>{riskLabel}</span>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Main grid: video + side panels */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_260px] gap-6 items-start">

          {/* Left: video + canvas */}
          <div className="flex flex-col gap-4">
            <div
              className="relative rounded-2xl overflow-hidden bg-[#0A0A0A] aspect-video shadow-xl border border-[#E0E0E0]"
              style={{ willChange: 'transform' }}
            >
              {/* Native video — plays at 60fps independently */}
              <video
                key={activeScenario}
                autoPlay loop muted playsInline preload="auto"
                className="absolute inset-0 w-full h-full object-cover"
                style={{ willChange: 'transform' }}
              >
                <source src={
                  activeScenario === 'sharp_turn' ? '/videos/output4.mp4' :
                  activeScenario === 'lane_change' ? '/videos/output3.mp4' :
                  activeScenario === 'urban' ? '/videos/output2.mp4' :
                  '/videos/output.mp4'
                } type="video/mp4" />
                <img
                  src="https://images.unsplash.com/photo-1555400038-63f5ba517a47?w=1280&q=80"
                  alt="fallback" className="absolute inset-0 w-full h-full object-cover opacity-60"
                />
              </video>

              {/* Transparent overlay canvas */}
              <canvas
                ref={canvasRef} width={640} height={360}
                className="absolute inset-0 w-full h-full z-10 pointer-events-none"
                style={{ willChange: 'transform' }}
              />

              {/* Corner reticles (CSS only) */}
              {['top-4 left-4 border-t border-l', 'top-4 right-4 border-t border-r',
                'bottom-4 left-4 border-b border-l', 'bottom-4 right-4 border-b border-r'
              ].map((cls, i) => (
                <div key={i} className={`absolute w-6 h-6 border-white/50 z-20 pointer-events-none rounded-sm ${cls}`} />
              ))}

              {/* Top HUD bar */}
              <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-4 py-2 bg-gradient-to-b from-black/60 to-transparent">
                <div className="flex items-center gap-2">
                  {isLive && <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />}
                  <span className="text-white text-xs font-semibold tracking-widest uppercase">
                    {isLive ? 'Live' : 'Offline'}
                  </span>
                  <span className="text-white/50 text-xs ml-1">· {activeSc.icon} {activeSc.label}</span>
                </div>
                <div className="flex items-center gap-3">
                  {isLive && (
                    <span className="bg-[#6366F1]/80 text-white text-[9px] font-bold px-2.5 py-0.5 rounded-full backdrop-blur-sm">
                      ⚛️ {activeModel}
                    </span>
                  )}
                  <span className="text-white/60 text-xs font-mono">
                    {t?.frame != null ? `Frame ${t.frame}` : '—'}
                  </span>
                </div>
              </div>

              {/* Offline overlay */}
              {wsStatus === 'error' && (
                <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-black/60">
                  <span className="text-white text-lg font-semibold">Simulation Offline</span>
                  <span className="text-white/60 text-sm">Start the FastAPI backend to connect</span>
                </div>
              )}

              {/* Bottom legend */}
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 flex gap-3">
                {[['#3B82F6','Car'],['#8B5CF6','Pedestrian'],['#F59E0B','Truck'],['rgba(99,102,241,0.7)','QNN Cone']].map(([c,l]) => (
                  <div key={l} className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full">
                    <div className="w-3 h-1 rounded" style={{ background: c }} />
                    <span className="text-white text-[10px] font-mono">{l}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom metric strip */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MetricCard label="Active Model" value={activeModel} sub="best performer" />
              <MetricCard label="FPS" value={fps} sub="render rate" />
              <MetricCard label="Latency" value={`${latency}ms`} sub="system latency" />
              <MetricCard label="QNN R²" value={modelR2} sub="model accuracy" />
              <MetricCard label="QNN ADE" value={ade} sub="displacement error" />
            </div>
          </div>

          {/* Right: risk panel + object list */}
          <div className="flex flex-col gap-4">
            <RiskPanel objects={objects} wsStatus={wsStatus} />

            {/* Object list */}
            {objects.length > 0 && (
              <div className="tesla-card">
                <div className="tesla-label text-[10px] mb-3">Tracked Objects ({objects.length})</div>
                <div className="flex flex-col gap-2">
                  {objects.slice(0, 5).map(obj => {
                    const clr = OBJECT_COLORS[obj.object_type ?? 'car'];
                    return (
                      <motion.div
                        key={obj.id}
                        layout
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-2.5 p-2 rounded-lg bg-[#FAFAFA] border border-[#F0F0F0]"
                      >
                        <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: clr }} />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-semibold text-[#171A20] capitalize truncate">
                            {obj.object_type ?? 'car'} #{obj.id}
                          </div>
                          <div className="text-[10px] text-[#5C5E62] truncate capitalize">{obj.behavior ?? 'moving'}</div>
                        </div>
                        <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full capitalize ${
                          obj.risk === 'danger' ? 'risk-danger' :
                          obj.risk === 'caution' ? 'risk-caution' : 'risk-safe'
                        }`}>{obj.risk}</span>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
