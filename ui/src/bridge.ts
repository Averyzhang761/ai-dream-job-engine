import { App } from "@modelcontextprotocol/ext-apps";

import { privatePayload, type ToolResultEnvelope } from "./contracts";

export const UI_UNAVAILABLE_MESSAGE =
  "UI unavailable; use the conversation tools";

type ToolResultListener = (result: ToolResultEnvelope) => void;

export interface McpAppBridge {
  initialize(): Promise<void>;
  callTool<T>(name: string, args: object): Promise<T>;
  sendMessage(message: string): Promise<void>;
  subscribeToToolResults(listener: ToolResultListener): () => void;
}

interface BridgeOptions {
  timeoutMs?: number;
}

export function createMcpAppBridge(
  { timeoutMs = 4_000 }: BridgeOptions = {},
): McpAppBridge {
  const app = new App(
    { name: "ai-dream-job-engine", version: "0.1.0" },
    {},
    { autoResize: true, strict: true },
  );
  const listeners = new Set<ToolResultListener>();
  let initialized = false;
  let initialization: Promise<void> | undefined;

  app.addEventListener("toolresult", (result) => {
    const envelope: ToolResultEnvelope = {
      structuredContent: result.structuredContent,
      _meta: result._meta,
      isError: result.isError,
    };
    for (const listener of listeners) {
      listener(envelope);
    }
  });

  const requireInitialized = () => {
    if (!initialized) {
      throw new Error(UI_UNAVAILABLE_MESSAGE);
    }
  };

  return {
    initialize() {
      if (initialization) {
        return initialization;
      }
      initialization = Promise.race([
        app.connect(),
        new Promise<never>((_, reject) => {
          window.setTimeout(
            () => reject(new Error(UI_UNAVAILABLE_MESSAGE)),
            timeoutMs,
          );
        }),
      ])
        .then(() => {
          initialized = true;
        })
        .catch(() => {
          throw new Error(UI_UNAVAILABLE_MESSAGE);
        });
      return initialization;
    },

    async callTool<T>(name: string, args: object): Promise<T> {
      requireInitialized();
      const result = await app.callServerTool({
        name,
        arguments: args as Record<string, unknown>,
      });
      const payload = privatePayload(result);
      if (result.isError || payload === undefined) {
        throw new Error("The local tool request failed.");
      }
      return payload as T;
    },

    async sendMessage(message: string): Promise<void> {
      requireInitialized();
      const result = await app.sendMessage({
        role: "user",
        content: [{ type: "text", text: message }],
      });
      if (result.isError) {
        throw new Error("The host Agent did not accept the request.");
      }
    },

    subscribeToToolResults(listener: ToolResultListener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
