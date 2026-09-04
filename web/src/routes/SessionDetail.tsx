import { useParams, Link } from "react-router-dom";
import { useState } from "react";
import {
  useSession,
  useSessionTrajectory,
  useSessionCompletions,
  useSessionEvaluation,
  useSessionRaw,
  useSessionTrace,
  useSessionArtifacts,
} from "../api/queries";
import { api } from "../api/client";
import { StageTimeline } from "../components/StageTimeline";
import { CompletionDiff } from "../components/CompletionDiff";
import { TraceList } from "../components/TraceList";
import { JsonView } from "../components/JsonView";
import { CopyBtn } from "../components/CopyBtn";
import { StatusPill } from "../components/StatusPill";
import { TraceWaterfall } from "../components/TraceWaterfall";
import { ArtifactList } from "../components/ArtifactList";
import { shortId } from "../utils";

type TabId = "timeline" | "trace" | "artifacts" | "completions" | "trajectory" | "evaluation" | "raw";

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [tab, setTab] = useState<TabId>("timeline");
  const session = useSession(sessionId, 2000);
  const trajectory = useSessionTrajectory(sessionId);
  const completions = useSessionCompletions(sessionId, 2500);
  const evaluation = useSessionEvaluation(sessionId);
  const raw = useSessionRaw(sessionId);
  const trace = useSessionTrace(sessionId);
  const artifacts = useSessionArtifacts(sessionId);

  const [cancelMsg, setCancelMsg] = useState<string | null>(null);
  const cancel = async () => {
    if (!sessionId) return;
    const ok = window.confirm("Cancel this session?");
    if (!ok) return;
    try {
      const res = await api.delete<any>(`/api/sessions/${sessionId}`);
      setCancelMsg(`Cancelled on ${res.node_id ?? "?"}`);
    } catch (err: any) {
      setCancelMsg(`Failed: ${err.message ?? String(err)}`);
    }
  };

  const data = session.data;
  const isRunning =
    data?.status && !["COMPLETED", "ERROR", "TIMEOUT"].includes(data.status);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Session</div>
            <div className="font-mono text-base">
              {data?.session_id ?? sessionId}
              {sessionId && <CopyBtn value={sessionId} />}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-slate-500">
              <StatusPill status={data?.status} />
              {data?.task_id && (
                <span>
                  task:{" "}
                  <Link
                    to={`/tasks/${data.task_id}`}
                    className="text-blue-600 hover:underline"
                  >
                    {shortId(data.task_id, 40)}
                  </Link>
                </span>
              )}
              {data?.node_id && <span>node: {data.node_id}</span>}
              {data?.source && <span>source: {data.source}</span>}
            </div>
            {data?.error && (
              <div className="mt-2 rounded bg-red-50 px-2 py-1 text-xs text-red-700">
                {data.error}
                <CopyBtn value={data.error} />
              </div>
            )}
          </div>
          {isRunning && (
            <button
              type="button"
              className="rounded border border-red-500 px-3 py-1 text-xs text-red-600 hover:bg-red-50"
              onClick={cancel}
            >
              Cancel
            </button>
          )}
        </div>
        {cancelMsg && <div className="mt-2 text-xs text-slate-500">{cancelMsg}</div>}
      </div>

      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="flex gap-1 border-b border-slate-200 px-2 py-1">
          {(
            [
              ["timeline", "Timeline"],
              ["trace", `Trace (${trace.data?.event_count ?? 0})`],
              ["artifacts", `Artifacts (${artifacts.data?.artifact_count ?? 0})`],
              ["completions", `Completions (${completions.data?.completions?.length ?? 0})`],
              ["trajectory", `Trajectory (${trajectory.data?.traces?.length ?? 0})`],
              ["evaluation", "Evaluation"],
              ["raw", "Raw JSON"],
            ] as [TabId, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`rounded px-3 py-1 text-sm ${
                tab === id
                  ? "bg-blue-100 text-blue-900"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="p-4">
          {tab === "timeline" && <StageTimeline timing={data?.timing} />}

          {tab === "trace" && (
            <TraceWaterfall
              trace={trace.data}
              loading={trace.isLoading}
              unavailable={trace.isError}
            />
          )}

          {tab === "artifacts" && (
            <ArtifactList payload={artifacts.data} loading={artifacts.isLoading} />
          )}

          {tab === "completions" && (
            <div className="space-y-2">
              <div className="text-xs text-slate-500">
                source: {completions.data?.source ?? "—"}
              </div>
              {(completions.data?.completions ?? []).map((c) => (
                <CompletionDiff key={c.completion_id} completion={c} />
              ))}
              {!completions.data?.completions?.length && (
                <div className="text-sm text-slate-500">
                  No completion records persisted for this session.
                </div>
              )}
            </div>
          )}

          {tab === "trajectory" && (
            <TraceList traces={trajectory.data?.traces ?? []} />
          )}

          {tab === "evaluation" && (
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded border border-slate-200 p-3">
                  <div className="text-xs uppercase text-slate-500">Outcome reward</div>
                  <div className="text-2xl font-semibold">
                    {evaluation.data?.outcome_reward ?? "—"}
                  </div>
                </div>
                <div className="rounded border border-slate-200 p-3">
                  <div className="text-xs uppercase text-slate-500">Strategy</div>
                  <div className="text-base">{evaluation.data?.strategy ?? "—"}</div>
                </div>
              </div>
              <JsonView value={evaluation.data?.raw ?? {}} maxHeight="320px" />
            </div>
          )}

          {tab === "raw" && (
            <div className="space-y-2 text-sm">
              {raw.data ? (
                <>
                  <div className="text-xs text-slate-500">file: {raw.data.file_path}</div>
                  <JsonView value={raw.data.data} maxHeight="540px" />
                </>
              ) : (
                <div className="text-slate-500">No raw session file on disk.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
