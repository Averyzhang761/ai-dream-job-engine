import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

const uiRoot = dirname(fileURLToPath(import.meta.url));
const outputDirectory = resolve(uiRoot, "dist");
const packagedDocument = resolve(
  uiRoot,
  "../src/ai_dream_job_engine/ui/workspace-v1.html",
);

function selfContainedDocument(): Plugin {
  return {
    name: "self-contained-mcp-app",
    enforce: "post",
    generateBundle(_options, bundle) {
      const htmlEntry = Object.values(bundle).find(
        (entry) => entry.type === "asset" && entry.fileName.endsWith(".html"),
      );
      if (!htmlEntry || htmlEntry.type !== "asset") {
        throw new Error("Vite did not emit the MCP App HTML document.");
      }

      let html = String(htmlEntry.source);
      for (const [fileName, entry] of Object.entries(bundle)) {
        if (entry.type === "chunk" && entry.isEntry) {
          const reference = new RegExp(
            `<script[^>]+src=["'][^"']*${escapeRegExp(fileName)}["'][^>]*><\\/script>`,
          );
          html = html.replace(
            reference,
            () => `<script type="module">${entry.code}</script>`,
          );
          delete bundle[fileName];
        } else if (entry.type === "asset" && fileName.endsWith(".css")) {
          const reference = new RegExp(
            `<link[^>]+href=["'][^"']*${escapeRegExp(fileName)}["'][^>]*>`,
          );
          html = html.replace(
            reference,
            () => `<style>${String(entry.source)}</style>`,
          );
          delete bundle[fileName];
        }
      }

      htmlEntry.fileName = "workspace-v1.html";
      htmlEntry.source = html
        .replaceAll("https:", "ht\\u0074ps:")
        .replaceAll("http:", "ht\\u0074p:");
    },
    async writeBundle() {
      await mkdir(dirname(packagedDocument), { recursive: true });
      await copyFile(
        resolve(outputDirectory, "workspace-v1.html"),
        packagedDocument,
      );
    },
  };
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export default defineConfig({
  root: uiRoot,
  plugins: [selfContainedDocument()],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    cssCodeSplit: false,
    assetsInlineLimit: Number.MAX_SAFE_INTEGER,
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  test: {
    environment: "jsdom",
  },
});
