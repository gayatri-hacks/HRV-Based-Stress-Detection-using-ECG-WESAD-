import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import ResultSlider from "./ResultSlider";

it("displays binary prediction and probability confidence", () => {
  render(<ResultSlider label="Stress Level" value={1} confidence={73} />);
  expect(screen.getAllByText("Stressed").length).toBeGreaterThan(0);
  expect(screen.getByText("73%")).toBeInTheDocument();
});
