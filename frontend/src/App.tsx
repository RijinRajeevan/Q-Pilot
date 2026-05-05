import React, { useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HeroSection from './sections/HeroSection';
import SimulationSection from './sections/SimulationSection';
import ScenariosSection from './sections/ScenariosSection';
import QuantumSection from './sections/QuantumSection';
import ModelComparisonSection from './sections/ModelComparisonSection';
import DatasetExplorerSection from './sections/DatasetExplorerSection';

function App() {
  const simRef = useRef<HTMLElement>(null);

  // Establish WebSocket connection globally
  useWebSocket();

  const scrollTo = (id: string) =>
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div className="min-h-screen bg-white text-[#171A20]">
      <Navbar
        onSim={() => scrollTo('simulation')}
        onTech={() => scrollTo('quantum')}
      />

      <main>
        <HeroSection
          onSimClick={() => scrollTo('simulation')}
          onExploreClick={() => scrollTo('scenarios')}
        />
        <SimulationSection sectionRef={simRef} />
        <ScenariosSection />
        <QuantumSection />
        <ModelComparisonSection />
        <DatasetExplorerSection />
      </main>

      <Footer />
    </div>
  );
}

export default App;
