import { Link } from "react-router-dom";
import { useTasks, useTopology } from "../api/queries";
import { TopologyGraph } from "../components/TopologyGraph";
import { StatusPill } from "../components/StatusPill";
import { relativeTime, shortId, formatReward } from "../utils";

export function Dashboard() {
  const topology = useTopology(2000);
  const tasks = useTasks(2000);

  const runningTasks = (tasks.data?.tasks ?? []).filter((t) => t.status === "running").length;
  const completedToday = (tasks.data?.tasks ?? []).filter((t) => {
    const ts = (t.updated_at ?? 0) * 1000;
    return Date.now() - ts < 86400_000;
  }).length;
  const recentTasks = (tasks.data?.tasks ?? []).slice(0, 10);

  return (
    <div className="space-y-6">
      <TopologyGraph topology={topology.data} isLoading={topology.isLoading} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Stat label="Running tasks" value={runningTasks} />
        <Stat label="Completed (24h)" value={completedToday} />
        <Stat label="Tracked tasks total" value={tasks.data?.tasks.length ?? 0} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2">
          <h2 className="text-sm font-medium">Recent tasks</h2>
          <Link to="/tasks" className="text-xs text-blue-600 hover:underline">
            view all →
          </Link>
        </div>
        {recentTasks.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-slate-500">
            No tasks tracked yet. Submit one via{" "}
            <code className="rounded bg-slate-100 px-1">polar submit</code> or the
            example scripts and it will appear here.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2">task_id</th>
                <th className="px-3 py-2">status</th>
                <th className="px-3 py-2">harness</th>
                <th className="px-3 py-2">reward (avg)</th>
                <th className="px-3 py-2">trace / completion</th>
                <th className="px-3 py-2">progress</th>
                <th className="px-3 py-2">updated</th>
              </tr>
            </thead>
            <tbody>
              {recentTasks.map((task) => (
                <tr key={task.task_id} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link to={`/tasks/${task.task_id}`} className="text-blue-600 hover:underline">
                      {shortId(task.task_id, 38)}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <StatusPill status={task.status} />
                  </td>
                  <td className="px-3 py-2 text-xs">{task.harness ?? "—"}</td>
                  <td className="px-3 py-2 text-xs font-mono">
                    {formatReward(task.mean_reward)}
                  </td>
                  <td className="px-3 py-2 text-xs font-mono">
                    {task.mean_traces != null ? task.mean_traces.toFixed(1) : "—"}
                    {" / "}
                    {task.mean_completions != null ? task.mean_completions.toFixed(1) : "—"}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {task.completed_sessions}/{task.num_samples}
                  </td>
                  <td className="px-3 py-2 text-xs text-slate-500">
                    {relativeTime(task.updated_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
    </div>
  );
}
