import { create } from 'zustand';

export type Scenario = 'highway' | 'lane_change' | 'urban' | 'emergency_brake' | 'sharp_turn';

export interface TrajectoryPoint {
  x: number;
  y: number;
  uncert: number;
}

export interface TrackedObject {
  id: number;
  current: { x: number; y: number };
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  confidence: number;
  behavior: string;
  risk: 'safe' | 'caution' | 'danger';
  object_type: 'car' | 'pedestrian' | 'truck' | 'cyclist';
  risk_reason?: string;
  suggested_action?: string;
  ttc?: number;
  speed?: number;
  velocity?: { vx: number; vy: number };
  acceleration?: { ax: number; ay: number };
  heading?: number;
  qnn: TrajectoryPoint[];
  final: TrajectoryPoint[];
  collision_warning: number[];
}

export interface SklearnMetrics {
  linear_r2: number;
  linear_mse: number;
  dt_r2: number;
  dt_mse: number;
  qnn_r2: number;
  qnn_mse: number;
  winner: string;
  winner_reason: string;
}

export interface Metrics {
  qnn: { ade: number; variance: number; latency: number };
  lstm: { ade: number; variance: number; latency: number };
}

export interface TelemetryFrame {
  frame: number;
  fps: number;
  system_latency: number;
  image: string | null;
  objects: TrackedObject[];
  metrics: Metrics;
  sklearn_metrics?: SklearnMetrics;
  ego?: { speed: number; acceleration: number };
  scenario: Scenario;
  logs: string[];
  // V7 fields
  pipeline?: string;
  best_model?: string;
  model_ranking?: string[];
  detection_count?: number;
  track_count?: number;
}

export type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface TelemetryState {
  telemetry: TelemetryFrame | null;
  wsStatus: WsStatus;
  retryCount: number;
  activeScenario: Scenario;
  setTelemetry: (t: TelemetryFrame) => void;
  setWsStatus: (s: WsStatus) => void;
  incrementRetry: () => void;
  resetRetry: () => void;
  setScenario: (s: Scenario) => void;
}

export const useTelemetryStore = create<TelemetryState>((set) => ({
  telemetry: null,
  wsStatus: 'connecting',
  retryCount: 0,
  activeScenario: 'highway',
  setTelemetry: (t) => set({ telemetry: t }),
  setWsStatus: (s) => set({ wsStatus: s }),
  incrementRetry: () => set((state) => ({ retryCount: state.retryCount + 1 })),
  resetRetry: () => set({ retryCount: 0 }),
  setScenario: (s) => set({ activeScenario: s }),
}));
