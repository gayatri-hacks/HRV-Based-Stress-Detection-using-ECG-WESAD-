import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import ResultSlider from "./ResultSlider";

it("labels a positive prediction with stress probability", () => {
  render(<ResultSlider label="Stress Level" value={1} probability={73} />);
  expect(screen.getAllByText("Stressed").length).toBeGreaterThan(0);
  expect(screen.getByText(/Stress probability/)).toBeInTheDocument();
  expect(screen.getByText("73%")).toBeInTheDocument();
});

it("labels a negative prediction without calling its probability confidence", () => {
  render(<ResultSlider label="Stress Level" value={0} probability={31} />);
  expect(screen.getAllByText("Not Stressed").length).toBeGreaterThan(0);
  expect(screen.getByText("31%")).toBeInTheDocument();
});
