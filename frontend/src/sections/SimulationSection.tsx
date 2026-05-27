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

// ─── Object colors ────────────────────────────────────────
const OBJECT_COLORS: Record<string, string> = {
  car:        '#3B82F6',
  pedestrian: '#8B5CF6',
  truck:      '#F59E0B',
  cyclist:    '#10B981',
};
const RISK_COLORS = { safe: '#059669', caution: '#D97706', danger: '#DC2626' };

// ─── Canvas overlay — draws real YOLO bounding boxes ──────
function drawOverlay(ctx: CanvasRenderingContext2D, objects: TrackedObject[], canvasW: number, canvasH: number) {
  if (!objects?.length) return;

  // Video resolution is 640x360, scale to canvas
  const scaleX = canvasW / 640;
  const scaleY = canvasH / 360;

  objects.forEach((obj) => {
    const bbox = obj.bbox;
    if (!bbox) return;

    const x1 = bbox.x1 * scaleX;
    const y1 = bbox.y1 * scaleY;
    const x2 = bbox.x2 * scaleX;
    const y2 = bbox.y2 * scaleY;
    const bw = x2 - x1;
    const bh = y2 - y1;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;

    const riskColor = RISK_COLORS[obj.risk ?? 'safe'];
    const objColor = OBJECT_COLORS[obj.object_type ?? 'car'] ?? '#3B82F6';

    // ── Bounding box rectangle ──
    ctx.strokeStyle = objColor;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.85;
    ctx.strokeRect(x1, y1, bw, bh);
    ctx.globalAlpha = 1.0;

    // ── Corner accents (Tesla-style) ──
    const cs = Math.min(bw, bh) * 0.25;
    ctx.lineWidth = 3;
    ctx.strokeStyle = riskColor;
    // Top-left
    ctx.beginPath(); ctx.moveTo(x1, y1 + cs); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cs, y1); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x2 - cs, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cs); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x1, y2 - cs); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cs, y2); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x2 - cs, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cs); ctx.stroke();

    // ── Label chip: class + confidence ──
    const typeLabel = (obj.object_type ?? 'car').toUpperCase();
    const confLabel = `${(obj.confidence ?? 0).toFixed(0)}%`;
    const chipText = `${typeLabel} ${confLabel}`;
    ctx.font = 'bold 10px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(chipText).width;
    const chipX = x1;
    const chipY = y1 - 20;

    // Background pill
    ctx.fillStyle = objColor;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.roundRect(chipX, chipY, tw + 12, 17, 3);
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(chipText, chipX + 6, chipY + 12);

    // ── Risk dot + TTC ──
    if (obj.risk === 'danger' || obj.risk === 'caution') {
      const ttcText = obj.ttc != null && obj.ttc < 99 ? `TTC ${obj.ttc.toFixed(1)}s` : '';
      if (ttcText) {
        ctx.font = 'bold 9px Inter, system-ui, sans-serif';
        const ttcW = ctx.measureText(ttcText).width;
        ctx.fillStyle = riskColor;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.roundRect(x2 - ttcW - 10, y1 - 20, ttcW + 8, 15, 3);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(ttcText, x2 - ttcW - 6, y1 - 9);
      }
    }

    // ── Future trajectory path (smooth curve) ──
    if (obj.final?.length > 1) {
      // Draw filled uncertainty cone first (behind the line)
      if (obj.final.some(p => (p.uncert ?? 0) > 0)) {
        ctx.beginPath();
        ctx.fillStyle = obj.risk === 'danger' ? 'rgba(220,38,38,0.08)' : 
                        obj.risk === 'caution' ? 'rgba(217,119,6,0.08)' : 
                        'rgba(59,130,246,0.06)';
        ctx.moveTo(cx, cy);
        // Right edge
        obj.final.forEach((pt) => {
          const ux = Math.max((pt.uncert ?? 0) * 0.4 * scaleX, 2);
          ctx.lineTo(pt.x * scaleX + ux, pt.y * scaleY);
        });
        // Left edge (reverse)
        for (let i = obj.final.length - 1; i >= 0; i--) {
          const pt = obj.final[i];
          const ux = Math.max((pt.uncert ?? 0) * 0.4 * scaleX, 2);
          ctx.lineTo(pt.x * scaleX - ux, pt.y * scaleY);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Draw trajectory line with fading opacity
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2.5;
      let prevX = cx, prevY = cy;
      obj.final.forEach((pt, idx) => {
        const alpha = 0.8 - (idx / obj.final.length) * 0.5;
        ctx.strokeStyle = riskColor;
        ctx.globalAlpha = Math.max(alpha, 0.2);
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        const px = pt.x * scaleX, py = pt.y * scaleY;
        ctx.lineTo(px, py);
        ctx.stroke();
        prevX = px;
        prevY = py;
      });
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;

      // Endpoint marker
      const lastPt = obj.final[obj.final.length - 1];
      ctx.beginPath();
      ctx.arc(lastPt.x * scaleX, lastPt.y * scaleY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = riskColor;
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }

    // ── Velocity arrow ──
    const vel = obj.velocity;
    if (vel && (Math.abs(vel.vx) > 0.1 || Math.abs(vel.vy) > 0.1)) {
      const arrowLen = Math.min(Math.sqrt(vel.vx*vel.vx + vel.vy*vel.vy) * 30, 40);
      const angle = Math.atan2(-vel.vy, vel.vx);
      const ax = cx + Math.cos(angle) * arrowLen;
      const ay = cy + Math.sin(angle) * arrowLen;
      ctx.strokeStyle = '#22D3EE';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      // Arrowhead
      const headLen = 6;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - headLen * Math.cos(angle - 0.4), ay - headLen * Math.sin(angle - 0.4));
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - headLen * Math.cos(angle + 0.4), ay - headLen * Math.sin(angle + 0.4));
      ctx.stroke();
      ctx.globalAlpha = 1.0;
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
function RiskPanel({ objects, wsStatus, telemetry }: { objects: TrackedObject[]; wsStatus: string; telemetry: any }) {
  const mostCritical = objects.find(o => o.risk === 'danger')
    ?? objects.find(o => o.risk === 'caution')
    ?? objects[0];

  const isLive = wsStatus === 'connected' || (telemetry?.objects != null);

  if (!isLive) {
    return (
      <div className="tesla-card flex flex-col items-center justify-center gap-3 min-h-[200px] text-center">
        <div className="w-10 h-10 rounded-full bg-[#F4F4F4] flex items-center justify-center text-xl">📡</div>
        <p className="text-sm text-[#5C5E62]">Connecting to Backend...</p>
        <p className="text-xs text-[#5C5E62]">Waiting for telemetry data</p>
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
  const objTypeIcon: Record<string, string> = { car:'🚗', pedestrian:'🚶', truck:'🚛', cyclist:'🚲' };

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

        {/* TTC */}
        {mostCritical.ttc != null && mostCritical.ttc < 99 && (
          <div>
            <div className="tesla-label text-[10px] mb-1">Time To Collision</div>
            <div className={`text-lg font-bold font-mono ${
              mostCritical.ttc < 1.5 ? 'text-red-600' :
              mostCritical.ttc < 3.0 ? 'text-amber-600' : 'text-emerald-600'
            }`}>
              {mostCritical.ttc.toFixed(1)}s
            </div>
          </div>
        )}

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
            <span className="tesla-label text-[10px]">Detection Confidence</span>
            <span className="font-mono font-bold text-[#171A20]">{(mostCritical.confidence ?? 0).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-[#E0E0E0] rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-[#3E6AE1] rounded-full"
              animate={{ width: `${mostCritical.confidence ?? 0}%` }}
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
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const t = telRef.current;
        if (t?.objects?.length) drawOverlay(ctx, t.objects, canvas.width, canvas.height);
      }
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
  const ade     = t?.metrics?.qnn?.ade?.toFixed(2) ?? '—';
  const objects = t?.objects ?? [];
  const trackCount = objects.length;

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
          <h2 className="tesla-h2 mt-2">Real-Time Motion Prediction</h2>
          <p className="tesla-body mt-2 max-w-lg">
            YOLOv8m detection → ByteTrack tracking → trajectory prediction → TTC risk analysis.
            Switch scenarios to see how the pipeline adapts.
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
              </video>

              {/* Transparent overlay canvas */}
              <canvas
                ref={canvasRef} width={640} height={360}
                className="absolute inset-0 w-full h-full z-10 pointer-events-none"
                style={{ willChange: 'transform' }}
              />

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
                  <span className="bg-white/15 text-white text-[9px] font-bold px-2.5 py-0.5 rounded-full backdrop-blur-sm">
                    {trackCount} objects
                  </span>
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
                {[['#3B82F6','Car'],['#8B5CF6','Person'],['#F59E0B','Truck'],['#10B981','Cyclist']].map(([c,l]) => (
                  <div key={l} className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full">
                    <div className="w-3 h-1 rounded" style={{ background: c }} />
                    <span className="text-white text-[10px] font-mono">{l}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom metric strip */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MetricCard label="Tracked" value={`${trackCount}`} sub="active objects" />
              <MetricCard label="FPS" value={fps} sub="render rate" />
              <MetricCard label="Latency" value={`${latency}ms`} sub="inference" />
              <MetricCard label="ADE" value={`${ade}px`} sub="displacement error" />
              <MetricCard label="Pipeline" value={t?.pipeline ?? 'YOLOv8s+ByteTrack'} sub="V10 optimized" />
            </div>
          </div>

          {/* Right: risk panel + object list */}
          <div className="flex flex-col gap-4">
            <RiskPanel objects={objects} wsStatus={wsStatus} telemetry={t} />

            {/* Object list */}
            {objects.length > 0 && (
              <div className="tesla-card">
                <div className="tesla-label text-[10px] mb-3">Tracked Objects ({objects.length})</div>
                <div className="flex flex-col gap-2">
                  {objects.slice(0, 6).map(obj => {
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
                        <div className="flex flex-col items-end gap-0.5">
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full capitalize ${
                            obj.risk === 'danger' ? 'risk-danger' :
                            obj.risk === 'caution' ? 'risk-caution' : 'risk-safe'
                          }`}>{obj.risk}</span>
                          <span className="text-[9px] text-[#5C5E62] font-mono">{(obj.confidence ?? 0).toFixed(0)}%</span>
                        </div>
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
