import {
  BriefcaseBusiness,
  ClipboardCheck,
  SlidersHorizontal,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  createMcpAppBridge,
  type McpAppBridge,
  UI_UNAVAILABLE_MESSAGE,
} from "./bridge";
import {
  isWorkspaceAppPayload,
  type WorkspaceAppPayload,
} from "./contracts";
import "./styles.css";

type ViewId = "jobs" | "standards" | "evidence";
type ConnectionState = "connecting" | "ready" | "unavailable";

const tabs = [
  { id: "jobs" as const, label: "岗位", icon: BriefcaseBusiness },
  { id: "standards" as const, label: "我的标准", icon: SlidersHorizontal },
  { id: "evidence" as const, label: "我的证据", icon: ClipboardCheck },
];

interface AppProps {
  bridge?: McpAppBridge;
}

export function App({ bridge: providedBridge }: AppProps) {
  const bridge = useMemo(
    () => providedBridge ?? createMcpAppBridge(),
    [providedBridge],
  );
  const [activeView, setActiveView] = useState<ViewId>("jobs");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [payload, setPayload] = useState<WorkspaceAppPayload | null>(null);
  const [isNarrow, setIsNarrow] = useState(() => window.innerWidth <= 480);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  useEffect(() => {
    const updateWidth = () => setIsNarrow(window.innerWidth <= 480);
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  useEffect(() => {
    let mounted = true;
    const unsubscribe = bridge.subscribeToToolResults((result) => {
      if (!mounted || result.isError) {
        return;
      }
      if (isWorkspaceAppPayload(result.structuredContent)) {
        setPayload(result.structuredContent);
        setActiveView(result.structuredContent.view === "jobs" ? "jobs" : "evidence");
      }
    });

    bridge
      .initialize()
      .then(() => {
        if (mounted) {
          setConnection("ready");
        }
      })
      .catch(() => {
        if (mounted) {
          setConnection("unavailable");
        }
      });

    return () => {
      mounted = false;
      unsubscribe();
    };
  }, [bridge]);

  const selectTab = (index: number) => {
    const nextTab = tabs[(index + tabs.length) % tabs.length];
    setActiveView(nextTab.id);
    window.requestAnimationFrame(() => tabRefs.current[index]?.focus());
  };

  const onTabKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let nextIndex: number | undefined;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (nextIndex === undefined) return;
    event.preventDefault();
    selectTab(nextIndex);
  };

  return (
    <main
      className={`app-shell${isNarrow ? " app-shell--narrow" : ""}`}
      data-testid="app-shell"
    >
      <header className="app-header">
        <p className="product-name">AI Dream Job Engine</p>
        <span
          className={`connection-dot connection-dot--${connection}`}
          aria-label={connection === "ready" ? "本地数据已连接" : "本地数据连接中"}
        />
      </header>

      <nav className="view-tabs" role="tablist" aria-label="工作区视图">
        {tabs.map(({ id, label, icon: Icon }, index) => (
          <button
            key={id}
            ref={(node) => {
              tabRefs.current[index] = node;
            }}
            id={`tab-${id}`}
            className="view-tab"
            type="button"
            role="tab"
            aria-controls={`panel-${id}`}
            aria-selected={activeView === id}
            tabIndex={activeView === id ? 0 : -1}
            onClick={() => setActiveView(id)}
            onKeyDown={(event) => onTabKeyDown(event, index)}
          >
            <Icon aria-hidden="true" size={16} strokeWidth={1.8} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <section
        id={`panel-${activeView}`}
        className="view-panel"
        role="tabpanel"
        aria-labelledby={`tab-${activeView}`}
      >
        <ViewContent
          activeView={activeView}
          connection={connection}
          payload={payload}
        />
      </section>

      <p className="sr-only" role="status" aria-live="polite">
        {connection === "unavailable"
          ? UI_UNAVAILABLE_MESSAGE
          : connection === "ready"
            ? "本地工作区已连接"
            : "正在连接本地工作区"}
      </p>
    </main>
  );
}

function ViewContent({
  activeView,
  connection,
  payload,
}: {
  activeView: ViewId;
  connection: ConnectionState;
  payload: WorkspaceAppPayload | null;
}) {
  if (connection === "unavailable") {
    return (
      <div className="state-message state-message--error">
        <strong>{UI_UNAVAILABLE_MESSAGE}</strong>
      </div>
    );
  }

  if (!payload) {
    return <div className="state-message">正在连接本地工作区…</div>;
  }

  if (activeView === "jobs") {
    const setupAction = payload.snapshot.setup_actions[0];
    if (setupAction) {
      return (
        <div className="state-message state-message--setup">
          {setupAction === "initialize_profile"
            ? "请先在对话中导入并确认简历证据。"
            : "请先在对话中初始化岗位标准。"}
        </div>
      );
    }
    return (
      <div className="state-message">
        {payload.snapshot.jobs.length === 0
          ? "尚无已分析岗位"
          : `已载入 ${payload.snapshot.jobs.length} 个岗位`}
      </div>
    );
  }

  if (activeView === "standards") {
    return (
      <div className="state-message">
        {payload.snapshot.rubric_ready ? "已载入当前标准" : "尚未初始化岗位标准"}
      </div>
    );
  }

  return (
    <div className="state-message">
      {payload.snapshot.profile_ready
        ? `已确认 ${payload.snapshot.confirmed_claims.length} 条证据`
        : "尚未建立候选人证据"}
    </div>
  );
}
