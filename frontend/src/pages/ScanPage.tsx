import { useCallback, useLayoutEffect, useState } from "react";
import { useParams, Navigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { useScanStore, type ScanType } from "@/store/useScanStore";
import Navbar from "@/components/Navbar";
import UploadZone from "@/components/UploadZone";
import ProcessingAnimation from "@/components/ProcessingAnimation";
import ResultDisplay from "@/components/ResultDisplay";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { z } from "zod";

const VALID_TYPES = ["ecg"];
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const stressResponseSchema = z.object({
  prediction: z.union([z.literal(0), z.literal(1)]),
  stress_probability: z.number().finite().min(0).max(1),
  predicted_class_confidence: z.number().finite().min(0).max(1),
  windows_used: z.number().int().positive(),
  minimum_duration_seconds: z.number().positive(),
  valid_windows_found: z.number().int().positive().optional(),
  selected_window: z.literal("final_valid").optional(),
  selected_window_start_seconds: z.number().min(0).optional(),
});
interface ApiError { detail?: string; }
const REQUEST_TIMEOUT_MS = 60_000;

const ScanPage = () => {
  const { type } = useParams<{ type: string }>();

  const file = useScanStore((s) => s.file);
  const isProcessing = useScanStore((s) => s.isProcessing);
  const result = useScanStore((s) => s.result);
  const startProcessing = useScanStore((s) => s.startProcessing);
  const setResult = useScanStore((s) => s.setResult);
  const setScanType = useScanStore((s) => s.setScanType);
  const setError = useScanStore((s) => s.setError);
  const error = useScanStore((s) => s.error);
  const currentScanType = useScanStore((s) => s.scanType);
  const [samplingRate, setSamplingRate] = useState("700");

  const isValid = VALID_TYPES.includes(type || "");
  const scanType = type as ScanType;

  useLayoutEffect(() => {
    if (isValid && currentScanType !== scanType) {
      setScanType(scanType);
    }
  }, [isValid, scanType, currentScanType, setScanType]);

  const handleFetch = useCallback(async () => {
    if (!file || isProcessing) return;
    if (Number(samplingRate) !== 700) {
      setError("This model is validated only for 700 Hz WESAD chest ECG recordings.");
      return;
    }

    startProcessing();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("fs", samplingRate);
      const response = await fetch(`${API_URL}/predict/ecg/stress`, { method: "POST", body: formData, signal: controller.signal });
      if (!response.ok) {
        const body = await response.json().catch(() => ({})) as ApiError;
        throw new Error(body.detail ?? `Backend request failed (${response.status})`);
      }
      const stressData = stressResponseSchema.parse(await response.json());

      setResult({
        stressLevel: stressData.prediction,
        stressProbability: Math.round(stressData.stress_probability * 100),

        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      setError(
        err instanceof DOMException && err.name === "AbortError"
          ? "The request timed out. Check that the API is available and try again."
          : err instanceof TypeError
            ? "The API is unavailable. Start the backend and try again."
            : err instanceof Error ? err.message : "Prediction request failed."
      );
    } finally {
      window.clearTimeout(timeout);
    }
  }, [file, isProcessing, samplingRate, startProcessing, setResult, setError]);

  if (!isValid) return <Navigate to="/" replace />;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto max-w-2xl px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10 text-center"
        >
          <span className="inline-block text-xs font-bold uppercase tracking-widest text-primary mb-2">
            {scanType.toUpperCase()} Analysis
          </span>
          <h1 className="text-3xl md:text-4xl font-extrabold font-display text-foreground">
            Upload Your ECG CSV
          </h1>
        </motion.div>

        <AnimatePresence mode="wait">
          {!isProcessing && !result && (
            <motion.div
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <UploadZone />

              <label className="block text-sm font-medium text-foreground" htmlFor="sampling-rate">
                Sampling rate (Hz)
                <input
                  id="sampling-rate"
                  type="number"
                  inputMode="numeric"
                  min="1"
                  step="1"
                  value={samplingRate}
                  onChange={(event) => setSamplingRate(event.target.value)}
                  className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2"
                  aria-describedby="sampling-rate-help"
                />
                <span id="sampling-rate-help" className="mt-1 block text-xs text-muted-foreground">Only 700 Hz WESAD chest ECG is supported by this trained model.</span>
              </label>

              {error && <p role="alert" className="text-sm text-destructive text-center">{error}</p>}

              <div className="flex justify-center">
                <Button
                  onClick={handleFetch}
                  disabled={!file || isProcessing}
                  className="gradient-primary text-primary-foreground font-semibold gap-2 px-8 py-5 text-base shadow-soft hover:shadow-elevated transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Fetch Results
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-center text-xs text-muted-foreground">Educational research output only; not a medical diagnosis or treatment recommendation.</p>
            </motion.div>
          )}

          {isProcessing && (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ProcessingAnimation />
            </motion.div>
          )}

          {result && !isProcessing && (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ResultDisplay />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default ScanPage;
