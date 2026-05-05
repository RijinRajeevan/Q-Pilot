import React, { memo } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';

const statusMap = {
  connected:    { dot: 'bg-emerald-500', label: 'System Online' },
  connecting:   { dot: 'bg-amber-400 animate-pulse', label: 'Connecting…' },
  disconnected: { dot: 'bg-amber-400 animate-pulse', label: 'Reconnecting…' },
  error:        { dot: 'bg-red-500', label: 'Offline' },
};

const Navbar = memo(function Navbar({ onSim, onTech }: { onSim: () => void; onTech: () => void }) {
  const wsStatus = useTelemetryStore(s => s.wsStatus);
  const s = statusMap[wsStatus] ?? statusMap.connecting;

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center justify-between px-6 md:px-12 bg-white/90 backdrop-blur-xl border-b border-[#E0E0E0]">
      {/* Logo */}
      <button
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        className="flex items-center gap-2.5 select-none"
      >
        <div className="w-8 h-8 rounded-lg bg-[#171A20] flex items-center justify-center">
          <span className="text-white font-bold text-sm tracking-tight">Q</span>
        </div>
        <span className="text-[#171A20] font-semibold text-base tracking-tight hidden sm:block">Q-Pilot</span>
      </button>

      {/* Nav links */}
      <div className="hidden md:flex items-center gap-8">
        {[
          ['Simulation', 'simulation'],
          ['Quantum', 'quantum'],
          ['Models', 'models'],
          ['Dataset', 'dataset'],
        ].map(([label, id]) => (
          <button
            key={id}
            onClick={() => scrollTo(id)}
            className="text-sm text-[#5C5E62] hover:text-[#171A20] transition-colors font-medium"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Status + CTA */}
      <div className="flex items-center gap-4">
        <div className="hidden sm:flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${s.dot}`} />
          <span className="text-xs text-[#5C5E62] font-mono">{s.label}</span>
        </div>
        <button onClick={onSim} className="btn-primary !py-2 !px-5 text-sm">
          Start Simulation
        </button>
      </div>
    </nav>
  );
});

export default Navbar;
