import { Link, useParams } from "react-router-dom";
import { useState } from "react";
import { useTask } from "../api/queries";
import { StatusPill } from "../components/StatusPill";
import { RewardChart } from "../components/RewardChart";
import { CopyBtn } from "../components/CopyBtn";
import { JsonView } from "../components/JsonView";
import { api } from "../api/client";
import { formatMs, formatReward, shortId } from "../utils";

export function TaskDetail() {
  const { taskId } = useParams<{ taskId: string }>();
  const { data, isLoading } = useTask(taskId, 2000);
  const [cancelling, setCancelling] = useState(false);
  const [cancelMessage, setCancelMessage] = useState<string | null>(null);

  if (isLoading) return <div className="text-sm text-slate-500">Loading…</div>;
  if (!data) return <div className="text-sm text-slate-500">No task data.</div>;

  const sessions: any[] = data.sessions ?? [];
  const rewards = sessions.map((s) => s.reward ?? null);

  const cancelAllRunning = async () => {
    const running = sessions.filter(
      (s) => !["COMPLETED", "ERROR", "TIMEOUT"].includes(s.status),
    );
    if (running.length === 0) {
      setCancelMessage("No running sessions to cancel.");
      return;
    }
    const confirmed = window.confirm(
      `Cancel ${running.length} running session(s)?`,
    );
    if (!confirmed) return;
    setCancelling(true);
    setCancelMessage(null);
    let ok = 0;
    let fail = 0;
    for (const s of running) {
      try {
        await api.delete(`/api/sessions/${s.session_id}`);
        ok += 1;
      } catch {
        fail += 1;
      }
    }
    setCancelling(false);
    setCancelMessage(`Cancelled ${ok}, failed ${fail}`);
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Task</div>
            <div className="font-mono text-base">
              {data.task_id}
              <CopyBtn value={data.task_id} />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-slate-500">
              <StatusPill status={data.status} />
              {data.harness && <span>harness: {data.harness}</span>}
              {data.model && <span>model: {data.model}</span>}
              {data.num_samples && (
                <span>
                  sessions: {data.completed_sessions ?? 0}/{data.num_samples}
                </span>
              )}
              <span>save_dir: {data.save_dir_path}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={cancelling}
              className="rounded border border-red-500 px-3 py-1 text-xs text-red-600 hover:bg-red-50 disabled:opacity-50"
              onClick={cancelAllRunning}
            >
              {cancelling ? "Cancelling…" : "Cancel all running"}
            </button>
          </div>
        </div>
        {cancelMessage && (
          <div className="mt-2 text-xs text-slate-500">{cancelMessage}</div>
        )}
      </div>

      <RewardChart rewards={rewards} />

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
        <div className="border-b border-slate-200 px-4 py-2 text-sm font-medium">
          Sessions ({sessions.length})
        </div>
        {sessions.length === 0 ? (
          <div className="px-4 py-6 text-center text-sm text-slate-500">
            No sessions recorded.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2">session_id</th>
                <th className="px-3 py-2">status</th>
                <th className="px-3 py-2">node</th>
                <th className="px-3 py-2">reward</th>
                <th className="px-3 py-2">init</th>
                <th className="px-3 py-2">run</th>
                <th className="px-3 py-2">postrun</th>
                <th className="px-3 py-2">error</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link
                      to={`/sessions/${s.session_id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {shortId(s.session_id, 26)}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <StatusPill status={s.status} />
                  </td>
                  <td className="px-3 py-2 text-xs">{s.node_id ?? "—"}</td>
                  <td className="px-3 py-2 font-mono text-xs">{formatReward(s.reward)}</td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {formatMs(s.timing?.init_ms)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {formatMs(s.timing?.run_ms)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {formatMs(s.timing?.postrun_ms)}
                  </td>
                  <td className="px-3 py-2 text-xs text-red-600">
                    {s.error ? shortId(s.error, 40) : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div>
        <details className="rounded-lg border border-slate-200 bg-white">
          <summary className="cursor-pointer px-4 py-2 text-sm font-medium">
            Raw task JSON
          </summary>
          <div className="px-4 pb-4">
            <JsonView value={data} collapsed={false} maxHeight="320px" />
          </div>
        </details>
      </div>
    </div>
  );
}
