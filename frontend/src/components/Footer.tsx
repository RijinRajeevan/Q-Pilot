import React, { memo } from 'react';

const Footer = memo(function Footer() {
  return (
    <footer className="bg-[#171A20] text-white py-12 px-6 md:px-12">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
        <div>
          <div className="flex items-center gap-2.5 mb-3">
            <div className="w-7 h-7 rounded-md bg-white flex items-center justify-center">
              <span className="text-[#171A20] font-bold text-sm">Q</span>
            </div>
            <span className="font-semibold text-base">Q-Pilot V4</span>
          </div>
          <p className="text-[#5C5E62] text-sm leading-relaxed max-w-xs">
            Quantum-enhanced autonomous driving simulation. Research demonstration.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-x-16 gap-y-2 text-sm text-[#5C5E62]">
          {['4-Qubit VQC', 'DeepSORT Tracking', 'Monte Carlo Dropout',
            'NGSIM Dataset', 'Linear Regression', 'YOLOv8n'].map(item => (
            <span key={item}>{item}</span>
          ))}
        </div>
        <p className="text-xs text-[#3E3F42] font-mono">
          Research Prototype · Not for production use
        </p>
      </div>
    </footer>
  );
});

export default Footer;
