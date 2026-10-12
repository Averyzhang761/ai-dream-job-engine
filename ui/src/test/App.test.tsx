import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { App } from "../App";
import type { McpAppBridge } from "../bridge";

function connectedBridge(): McpAppBridge {
  return {
    initialize: async () => undefined,
    callTool: async <T,>() => ({}) as T,
    sendMessage: async () => undefined,
    subscribeToToolResults: () => () => undefined,
  };
}

describe("App shell", () => {
  afterEach(() => {
    cleanup();
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1024 });
  });

  it("renders the three primary views with jobs selected by default", () => {
    render(<App bridge={connectedBridge()} />);

    expect(screen.getByRole("tab", { name: "岗位" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "我的标准" })).toBeVisible();
    expect(screen.getByRole("tab", { name: "我的证据" })).toBeVisible();
  });

  it("switches tabs with arrow keys", async () => {
    const user = userEvent.setup();
    render(<App bridge={connectedBridge()} />);

    const jobs = screen.getByRole("tab", { name: "岗位" });
    jobs.focus();
    await user.keyboard("{ArrowRight}");

    expect(screen.getByRole("tab", { name: "我的标准" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("keeps every tab related to a present panel and hides inactive panels", () => {
    render(<App bridge={connectedBridge()} />);

    for (const tab of screen.getAllByRole("tab")) {
      const panelId = tab.getAttribute("aria-controls");
      expect(panelId).toBeTruthy();
      expect(document.getElementById(panelId!)).not.toBeNull();
    }
    expect(screen.getByRole("tabpanel", { name: "岗位" })).toBeVisible();
    expect(document.getElementById("panel-standards")).toHaveAttribute("hidden");
    expect(document.getElementById("panel-evidence")).toHaveAttribute("hidden");
  });

  it("marks a 360px viewport as a stable narrow panel", () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 360 });
    render(<App bridge={connectedBridge()} />);

    expect(screen.getByTestId("app-shell")).toHaveClass("app-shell--narrow");
  });

  it("shows the fixed fallback when the host bridge is unavailable", async () => {
    const bridge = connectedBridge();
    bridge.initialize = async () => {
      throw new Error("host detail must not be shown");
    };

    render(<App bridge={bridge} />);

    const activePanel = await screen.findByRole("tabpanel");
    expect(
      within(activePanel).getByText("UI unavailable; use the conversation tools"),
    ).toBeVisible();
    expect(screen.queryByText("host detail must not be shown")).not.toBeInTheDocument();
  });
});
