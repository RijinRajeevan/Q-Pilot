import { useEffect, useRef } from 'react';
import { useTelemetryStore } from '../store/telemetryStore';
import type { Scenario } from '../store/telemetryStore';

const BASE_WS = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/telemetry';
const MAX_RETRIES = 10;
const BASE_RETRY_DELAY = 1000;
const FRAME_INTERVAL_MS = 66; // ~15 FPS cap on React rerenders

export function useWebSocket() {
  const activeScenario = useTelemetryStore((s) => s.activeScenario);
  
  const wsRef = useRef<WebSocket | null>(null);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const lastFrameRef = useRef<number>(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (retryTimer.current) clearTimeout(retryTimer.current);
      if (wsRef.current) {
        wsRef.current.close(1000);
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const connect = () => {
      if (!mountedRef.current) return;
      if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return;

      const store = useTelemetryStore.getState();
      
      store.setWsStatus('connecting');
      const url = `${BASE_WS}?scenario=${activeScenario}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        useTelemetryStore.getState().setWsStatus('connected');
        useTelemetryStore.getState().resetRetry();
      };

      ws.onmessage = (e) => {
        if (!mountedRef.current) return;
        const now = performance.now();
        if (now - lastFrameRef.current < FRAME_INTERVAL_MS) return;
        lastFrameRef.current = now;
        try {
          const data = JSON.parse(e.data);
          const s = useTelemetryStore.getState();
          s.setTelemetry({ ...data, scenario: data.scenario ?? s.activeScenario });
        } catch { /* ignore malformed frames */ }
      };

      ws.onclose = (e) => {
        if (!mountedRef.current) return;
        const s = useTelemetryStore.getState();
        s.setWsStatus('disconnected');
        if (wsRef.current === ws) wsRef.current = null;
        
        if (e.code === 1000) return;

        const retry = s.retryCount + 1;
        if (retry <= MAX_RETRIES) {
          s.incrementRetry();
          const delay = Math.min(BASE_RETRY_DELAY * 2 ** (retry - 1), 30_000);
          retryTimer.current = setTimeout(connect, delay);
        } else {
          s.setWsStatus('error');
        }
      };

      ws.onerror = () => ws.close();
    };

    if (wsRef.current) {
      wsRef.current.close(1000);
      wsRef.current = null;
    }
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    useTelemetryStore.getState().resetRetry();
    
    const timer = setTimeout(connect, 300);  // Allow old WS to fully close
    return () => clearTimeout(timer);
  }, [activeScenario]);
}
