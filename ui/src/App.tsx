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
import { EvidenceView } from "./components/EvidenceView";
import { RubricView } from "./components/RubricView";
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
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const [decisionRefresh, setDecisionRefresh] = useState(0);
  const [staleJobIds, setStaleJobIds] = useState<Set<string>>(() => new Set());
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
          {activeView === id && (
            <ViewContent
              activeView={id}
              connection={connection}
              payload={payload}
              bridge={bridge}
              selectedAnalysisId={selectedAnalysisId}
              onSelectedAnalysisChange={setSelectedAnalysisId}
              decisionRefresh={decisionRefresh}
              staleJobIds={staleJobIds}
              onRubricSaved={() => {
                setStaleJobIds(
                  new Set(payload?.snapshot.jobs.map((job) => job.analysis_id) ?? []),
                );
              }}
              onDecisionRecomputed={(analysisId) => {
                setStaleJobIds((current) => {
                  const next = new Set(current);
                  next.delete(analysisId);
                  return next;
                });
                setDecisionRefresh((value) => value + 1);
              }}
            />
          )}
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
  selectedAnalysisId,
  onSelectedAnalysisChange,
  decisionRefresh,
  staleJobIds,
  onRubricSaved,
  onDecisionRecomputed,
}: {
  activeView: ViewId;
  connection: ConnectionState;
  payload: WorkspaceAppPayload | null;
  bridge: McpAppBridge;
  selectedAnalysisId: string | null;
  onSelectedAnalysisChange: (analysisId: string | null) => void;
  decisionRefresh: number;
  staleJobIds: Set<string>;
  onRubricSaved: () => void;
  onDecisionRecomputed: (analysisId: string) => void;
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
    return <JobWorkspace bridge={bridge} initialJobs={payload.snapshot.jobs} decisionRefresh={decisionRefresh} staleJobIds={staleJobIds} onSelectedAnalysisChange={onSelectedAnalysisChange} />;
  }

  if (activeView === "standards") {
    return payload.snapshot.rubric_ready ? (
      <RubricView
        bridge={bridge}
        selectedAnalysisId={selectedAnalysisId}
        staleJobCount={staleJobIds.size}
        selectedJobIsStale={Boolean(selectedAnalysisId && staleJobIds.has(selectedAnalysisId))}
        onSaved={onRubricSaved}
        onRecomputed={onDecisionRecomputed}
      />
    ) : <div className="state-message">尚未初始化岗位标准</div>;
  }

  return payload.snapshot.profile_ready ? (
    <EvidenceView
      bridge={bridge}
      initialClaims={payload.snapshot.confirmed_claims}
      claimPreview={payload.claim_preview}
    />
  ) : <div className="state-message">尚未建立候选人证据</div>;
}

function JobWorkspace({
  bridge,
  initialJobs,
  decisionRefresh,
  staleJobIds,
  onSelectedAnalysisChange,
}: {
  bridge: McpAppBridge;
  initialJobs: WorkspaceJobSummary[];
  decisionRefresh: number;
  staleJobIds: Set<string>;
  onSelectedAnalysisChange: (analysisId: string | null) => void;
}) {
  const [jobs, setJobs] = useState(initialJobs);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialJobs[0]?.analysis_id ?? null,
  );
  const [detail, setDetail] = useState<WorkspaceJobDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(Boolean(selectedId));
  const [busy, setBusy] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const [detailLoadFailed, setDetailLoadFailed] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const selectedIdRef = useRef(selectedId);

  useEffect(() => {
    setJobs(initialJobs);
    setSelectedId((current) => {
      const next =
        current && initialJobs.some((job) => job.analysis_id === current)
          ? current
          : initialJobs[0]?.analysis_id ?? null;
      selectedIdRef.current = next;
      onSelectedAnalysisChange(next);
      return next;
    });
  }, [initialJobs, onSelectedAnalysisChange]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setLoadingDetail(false);
      return;
    }
    let cancelled = false;
    const requestedId = selectedId;
    setLoadingDetail(true);
    setDetailLoadFailed(false);
    bridge
      .callTool<WorkspaceJobDetail>("get_workspace_job", {
        analysis_id: selectedId,
      })
      .then((nextDetail) => {
        if (
          !cancelled &&
          nextDetail.summary.analysis_id === requestedId &&
          selectedIdRef.current === requestedId
        ) {
          setDetail(nextDetail);
        }
      })
      .catch(() => {
        if (!cancelled && selectedIdRef.current === requestedId) {
          setDetail(null);
          setDetailLoadFailed(true);
          setAnnouncement("无法载入岗位详情；请重试");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [bridge, decisionRefresh, loadAttempt, selectedId]);

  const replaceFromServer = (
    requestedId: string,
    nextDetail: WorkspaceJobDetail,
  ) => {
    if (nextDetail.summary.analysis_id !== requestedId) return false;
    setJobs((current) =>
      current.map((job) =>
        job.analysis_id === nextDetail.summary.analysis_id
          ? nextDetail.summary
          : job,
      ),
    );
    if (selectedIdRef.current === requestedId) {
      setDetail(nextDetail);
    }
    return true;
  };

  const refreshJob = async (analysisId: string) => {
    try {
      const refreshed = await bridge.callTool<WorkspaceJobDetail>(
        "get_workspace_job",
        { analysis_id: analysisId },
      );
      return replaceFromServer(analysisId, refreshed);
    } catch {
      return false;
    }
  };

  const review = async (status: JobReviewStatus) => {
    const actionAnalysisId = detail?.summary.analysis_id;
    if (
      !actionAnalysisId ||
      selectedIdRef.current !== actionAnalysisId ||
      busy
    ) return;
    setBusy(true);
    setAnnouncement("");
    try {
      const nextDetail = await bridge.callTool<WorkspaceJobDetail>(
        "set_job_review_status",
        { analysis_id: actionAnalysisId, status },
      );
      const accepted = replaceFromServer(actionAnalysisId, nextDetail);
      if (accepted && selectedIdRef.current === actionAnalysisId) {
        setAnnouncement(
          status === "kept"
            ? "已保留此岗位"
            : status === "ignored"
              ? "已忽略此岗位"
              : "已加入待申请队列",
        );
      }
    } catch {
      const refreshed = await refreshJob(actionAnalysisId);
      if (selectedIdRef.current === actionAnalysisId) {
        setAnnouncement(
          refreshed
            ? "状态未更新；已刷新服务端结果"
            : "状态未更新；请重试",
        );
      }
    } finally {
      setBusy(false);
    }
  };

  const selectJob = (analysisId: string) => {
    selectedIdRef.current = analysisId;
    setSelectedId(analysisId);
    onSelectedAnalysisChange(analysisId);
    setDetail(null);
    setAnnouncement("");
  };

  if (jobs.length === 0) {
    return <div className="state-message">尚无已分析岗位</div>;
  }

  return (
    <div className="job-workspace-stack">
      {staleJobIds.size > 0 && (
        <p className="stale-warning" role="status">
          标准已更改：{staleJobIds.size} 个岗位仍显示旧结果
        </p>
      )}
      <div className="job-workspace">
      <aside className="job-queue" aria-label="岗位队列区域">
        <div className="section-heading">
          <h2>岗位</h2>
          <span>{jobs.length}</span>
        </div>
        <JobList jobs={jobs} selectedId={selectedId} onSelect={selectJob} />
      </aside>
      <div className="job-detail-region">
        {loadingDetail ? (
          <div className="state-message">正在载入岗位判断…</div>
        ) : detailLoadFailed ? (
          <div className="state-message detail-error">
            <p>无法载入岗位详情</p>
            <button
              type="button"
              onClick={() => {
                setAnnouncement("");
                setLoadAttempt((current) => current + 1);
              }}
            >
              重试载入详情
            </button>
          </div>
        ) : detail ? (
          <>
            {staleJobIds.has(detail.summary.analysis_id) && (
              <p className="stale-detail-warning">
                此岗位结果基于旧标准，请先重新计算。
              </p>
            )}
            <JobDetail detail={detail} busy={busy} onReview={review} />
          </>
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
    </div>
  );
}
