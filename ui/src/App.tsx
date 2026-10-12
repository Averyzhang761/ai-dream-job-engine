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
import { JobDetail } from "./components/JobDetail";
import { JobList } from "./components/JobList";
import {
  isWorkspaceAppPayload,
  type JobReviewStatus,
  type WorkspaceAppPayload,
  type WorkspaceJobDetail,
  type WorkspaceJobSummary,
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

      {tabs.map(({ id }) => (
        <section
          key={id}
          id={`panel-${id}`}
          className="view-panel"
          role="tabpanel"
          aria-labelledby={`tab-${id}`}
          hidden={activeView !== id}
        >
          <ViewContent
            activeView={id}
            connection={connection}
            payload={payload}
            bridge={bridge}
          />
        </section>
      ))}

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
  bridge,
}: {
  activeView: ViewId;
  connection: ConnectionState;
  payload: WorkspaceAppPayload | null;
  bridge: McpAppBridge;
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
    return <JobWorkspace bridge={bridge} initialJobs={payload.snapshot.jobs} />;
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

function JobWorkspace({
  bridge,
  initialJobs,
}: {
  bridge: McpAppBridge;
  initialJobs: WorkspaceJobSummary[];
}) {
  const [jobs, setJobs] = useState(initialJobs);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialJobs[0]?.analysis_id ?? null,
  );
  const [detail, setDetail] = useState<WorkspaceJobDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(Boolean(selectedId));
  const [busy, setBusy] = useState(false);
  const [announcement, setAnnouncement] = useState("");

  useEffect(() => {
    setJobs(initialJobs);
    setSelectedId((current) =>
      current && initialJobs.some((job) => job.analysis_id === current)
        ? current
        : initialJobs[0]?.analysis_id ?? null,
    );
  }, [initialJobs]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setLoadingDetail(false);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    bridge
      .callTool<WorkspaceJobDetail>("get_workspace_job", {
        analysis_id: selectedId,
      })
      .then((nextDetail) => {
        if (!cancelled) setDetail(nextDetail);
      })
      .catch(() => {
        if (!cancelled) {
          setDetail(null);
          setAnnouncement("无法载入岗位详情；请重试");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [bridge, selectedId]);

  const replaceFromServer = (nextDetail: WorkspaceJobDetail) => {
    setDetail(nextDetail);
    setJobs((current) =>
      current.map((job) =>
        job.analysis_id === nextDetail.summary.analysis_id
          ? nextDetail.summary
          : job,
      ),
    );
  };

  const refreshSelected = async () => {
    if (!selectedId) return false;
    try {
      const refreshed = await bridge.callTool<WorkspaceJobDetail>(
        "get_workspace_job",
        { analysis_id: selectedId },
      );
      replaceFromServer(refreshed);
      return true;
    } catch {
      return false;
    }
  };

  const review = async (status: JobReviewStatus) => {
    if (!selectedId || busy) return;
    setBusy(true);
    setAnnouncement("");
    try {
      const nextDetail = await bridge.callTool<WorkspaceJobDetail>(
        "set_job_review_status",
        { analysis_id: selectedId, status },
      );
      replaceFromServer(nextDetail);
      setAnnouncement(
        status === "kept"
          ? "已保留此岗位"
          : status === "ignored"
            ? "已忽略此岗位"
            : "已加入待申请队列",
      );
    } catch {
      const refreshed = await refreshSelected();
      setAnnouncement(
        refreshed
          ? "状态未更新；已刷新服务端结果"
          : "状态未更新；请重试",
      );
    } finally {
      setBusy(false);
    }
  };

  if (jobs.length === 0) {
    return <div className="state-message">尚无已分析岗位</div>;
  }

  return (
    <div className="job-workspace">
      <aside className="job-queue" aria-label="岗位队列区域">
        <div className="section-heading">
          <h2>岗位</h2>
          <span>{jobs.length}</span>
        </div>
        <JobList jobs={jobs} selectedId={selectedId} onSelect={setSelectedId} />
      </aside>
      <div className="job-detail-region">
        {loadingDetail ? (
          <div className="state-message">正在载入岗位判断…</div>
        ) : detail ? (
          <JobDetail detail={detail} busy={busy} onReview={review} />
        ) : (
          <div className="state-message">请选择岗位</div>
        )}
        {announcement && (
          <p className="workspace-announcement" role="status" aria-live="polite">
            {announcement}
          </p>
        )}
      </div>
    </div>
  );
}
