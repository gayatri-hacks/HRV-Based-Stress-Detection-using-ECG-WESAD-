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

it("sends the entered supported sampling rate in multipart form data", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ prediction: 1, stress_probability: 0.7, predicted_class_confidence: 0.7, windows_used: 1, minimum_duration_seconds: 60 }),
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<MemoryRouter initialEntries={["/scan/ecg"]}><Routes><Route path="/scan/:type" element={<ScanPage />} /></Routes></MemoryRouter>);
  await screen.findByLabelText(/sampling rate/i);
  fireEvent.change(document.querySelector("input[type=file]") as HTMLInputElement, { target: { files: [new File(["signal\\n0.1\\n"], "wesad.csv", { type: "text/csv" })] } });
  await screen.findByText("wesad.csv");
  fireEvent.change(screen.getByLabelText(/sampling rate/i), { target: { value: "700.0" } });
  fireEvent.click(screen.getByRole("button", { name: /fetch results/i }));
  await screen.findByText("Analysis Complete");
  const request = fetchMock.mock.calls[0][1] as RequestInit;
  expect((request.body as FormData).get("fs")).toBe("700.0");
});

it("rejects an unsupported sampling rate before a request", async () => {
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  render(<MemoryRouter initialEntries={["/scan/ecg"]}><Routes><Route path="/scan/:type" element={<ScanPage />} /></Routes></MemoryRouter>);
  await screen.findByLabelText(/sampling rate/i);
  fireEvent.change(document.querySelector("input[type=file]") as HTMLInputElement, { target: { files: [new File(["signal\\n0.1\\n"], "wesad.csv", { type: "text/csv" })] } });
  await screen.findByText("wesad.csv");
  fireEvent.change(screen.getByLabelText(/sampling rate/i), { target: { value: "250" } });
  fireEvent.click(screen.getByRole("button", { name: /fetch results/i }));
  expect(await screen.findByRole("alert")).toHaveTextContent("validated only for 700 Hz");
  expect(fetchMock).not.toHaveBeenCalled();
});
