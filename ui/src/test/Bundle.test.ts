import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const packagedHtml = resolve(
  process.cwd(),
  "src/ai_dream_job_engine/ui/workspace-v1.html",
);

describe("packaged MCP App", () => {
  it("executes the inline bundle and renders the app shell", async () => {
    const html = readFileSync(packagedHtml, "utf8");
    const inlineScript = html.match(
      /<script type="module">([\s\S]*?)<\/script>/,
    );
    expect(inlineScript).not.toBeNull();
    const executableHtml = html
      .replace(inlineScript![0], "")
      .replace(/<style>[\s\S]*?<\/style>/, "")
      .replace(
        "</body>",
        () => `<script>${inlineScript![1]}</script></body>`,
      );
    const errors: Error[] = [];
    const dom = new JSDOM(executableHtml, {
      runScripts: "dangerously",
      beforeParse(window) {
        window.addEventListener("error", (event) => {
          errors.push(event.error ?? new Error(event.message));
        });
      },
    });

    await new Promise((resolveTick) => dom.window.setTimeout(resolveTick, 25));

    expect(errors).toEqual([]);
    expect(
      dom.window.document.querySelector('[data-testid="app-shell"]'),
    ).not.toBeNull();
    expect(
      Array.from(dom.window.document.querySelectorAll('[role="tab"]')).map(
        (tab) => tab.textContent,
      ),
    ).toEqual(["岗位", "筛选标准", "我的证据"]);
    dom.window.close();
  });
});
