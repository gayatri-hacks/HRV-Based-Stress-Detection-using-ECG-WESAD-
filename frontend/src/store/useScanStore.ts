import { create } from "zustand";

export type ScanType = "ecg";

interface ScanResult {
  stressLevel: number;
  stressConfidence: number;   // %
  timestamp: string;
}

interface ScanState {
  scanType: ScanType | null;
  file: File | null;
  isProcessing: boolean;
  result: ScanResult | null;
  error: string | null;

  setScanType: (type: ScanType) => void;
  setFile: (file: File | null) => void;
  startProcessing: () => void;
  setResult: (result: ScanResult) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useScanStore = create<ScanState>((set) => ({
  scanType: null,
  file: null,
  isProcessing: false,
  result: null,
  error: null,

  setScanType: (type) =>
    set({
      scanType: type,
      file: null,
      result: null,
      isProcessing: false, error: null,
    }),

  setFile: (file) => set({ file, error: null, result: null }),

  startProcessing: () =>
    set({
      isProcessing: true,
      result: null,
    }),

  // 🔥 Important: stops processing automatically
  setResult: (result) =>
    set({
      result,
      isProcessing: false,
    }),
  setError: (error) => set({ error, isProcessing: false }),

  reset: () =>
    set({
      scanType: null,
      file: null,
      isProcessing: false,
      result: null,
      error: null,
    }),
}));
