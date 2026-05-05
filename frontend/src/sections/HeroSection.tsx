import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const STATS = [
  { value: '99.2%', label: 'Tracking Accuracy' },
  { value: '3.2×',  label: 'Over Classical LSTM' },
  { value: '4',     label: 'Qubits in Circuit' },
  { value: '<30ms', label: 'Inference Latency' },
];

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 32 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } },
};

export default function HeroSection({
  onSimClick,
  onExploreClick,
}: {
  onSimClick: () => void;
  onExploreClick: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = 0.85; // subtle slow-motion feel
    }
  }, []);

  return (
    <section
      id="hero"
      className="relative w-full min-h-screen flex flex-col items-center justify-center overflow-hidden"
    >
      {/* ── Background video ── */}
      <div className="absolute inset-0">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          className="absolute inset-0 w-full h-full object-cover"
          style={{ willChange: 'transform' }}
        >
          <source src="/videos/output.mp4" type="video/mp4" />
          {/* Fallback */}
          <img
            src="https://images.unsplash.com/photo-1555400038-63f5ba517a47?w=1920&q=80"
            alt="Autonomous driving highway"
            className="absolute inset-0 w-full h-full object-cover"
          />
        </video>
        {/* Cinematic gradient: stronger at bottom for text */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/50 via-black/20 to-black/80" />
      </div>

      {/* ── Content ── */}
      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="relative z-10 max-w-5xl mx-auto text-center px-6 pt-20"
      >
        {/* Badge */}
        <motion.div variants={fadeUp} className="inline-flex items-center gap-2 mb-7">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] uppercase tracking-[0.2em] font-semibold text-white/80">
            Quantum Intelligence Active
          </span>
        </motion.div>

        {/* H1 */}
        <motion.h1
          variants={fadeUp}
          className="text-white font-bold leading-[1.0] tracking-tight mb-6"
          style={{ fontSize: 'clamp(2.8rem, 7vw, 6rem)' }}
        >
          Quantum-Enhanced<br />
          <span
            className="text-transparent"
            style={{ WebkitTextStroke: '1px rgba(255,255,255,0.9)' }}
          >
            Autonomous
          </span>{' '}
          Driving
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          variants={fadeUp}
          className="text-white/70 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed mb-10"
        >
          Real-time trajectory prediction powered by a 4-qubit Variational Quantum Circuit
          and Monte Carlo uncertainty modeling — running live on your hardware.
        </motion.p>

        {/* CTAs */}
        <motion.div variants={fadeUp} className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20">
          <button onClick={onSimClick} className="btn-primary">
            View Live Simulation
          </button>
          <button onClick={onExploreClick} className="btn-secondary">
            Explore Technology
          </button>
        </motion.div>

        {/* Stats strip */}
        <motion.div
          variants={fadeUp}
          className="grid grid-cols-2 md:grid-cols-4 gap-px bg-white/10 rounded-2xl overflow-hidden backdrop-blur-sm border border-white/10"
        >
          {STATS.map(({ value, label }) => (
            <div key={label} className="bg-white/5 px-6 py-5 text-center">
              <div className="text-2xl md:text-3xl font-bold text-white tracking-tight">{value}</div>
              <div className="text-[11px] text-white/50 uppercase tracking-widest mt-1 font-medium">{label}</div>
            </div>
          ))}
        </motion.div>
      </motion.div>

      {/* Scroll cue */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 cursor-pointer"
        onClick={() => document.getElementById('simulation')?.scrollIntoView({ behavior: 'smooth' })}
      >
        <span className="text-[10px] uppercase tracking-[0.2em] text-white/50">Scroll</span>
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.8 }}
          className="w-5 h-5 border border-white/30 rounded-full flex items-center justify-center"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeOpacity="0.6">
            <path d="M6 9l6 6 6-6"/>
          </svg>
        </motion.div>
      </motion.div>
    </section>
  );
}
