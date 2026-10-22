import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("starts as one full-screen button in Indonesian", () => {
    render(<App />);
    const start = screen.getByRole("button", { name: /Ketuk di mana saja untuk mulai/ });
    expect(start).toHaveClass("tap-area");
    expect(screen.getByRole("main")).toHaveAttribute("lang", "id");
  });

  it("hides the camera picture from screen readers and has a live region", () => {
    const { container } = render(<App />);
    expect(container.querySelector("video")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "assertive");
  });

  it("switches to English", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("button", { name: /Tap anywhere to start/ })).toBeInTheDocument();
    expect(screen.getByRole("main")).toHaveAttribute("lang", "en");
  });
});
