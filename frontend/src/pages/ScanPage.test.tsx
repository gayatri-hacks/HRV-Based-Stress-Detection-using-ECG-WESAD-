import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import ScanPage from "./ScanPage";
import { useScanStore } from "@/store/useScanStore";

afterEach(() => {
  vi.unstubAllGlobals();
  useScanStore.getState().reset();
});

it("uploads an ECG CSV and renders the API stress result", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ prediction: 0, stress_probability: 0.3069808669, predicted_class_confidence: 0.6930191331, windows_used: 1, minimum_duration_seconds: 60 }),
  }));
  render(<MemoryRouter initialEntries={["/scan/ecg"]}><Routes><Route path="/scan/:type" element={<ScanPage />} /></Routes></MemoryRouter>);
  const input = document.querySelector("input[type=file]") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["signal\n0.1\n"], "wesad.csv", { type: "text/csv" })] } });
  fireEvent.click(await screen.findByRole("button", { name: /fetch results/i }));
  expect(await screen.findByText("Analysis Complete")).toBeInTheDocument();
  expect(screen.getAllByText("Not Stressed").length).toBeGreaterThan(0);
  expect(screen.getByText("31%")).toBeInTheDocument();
});
