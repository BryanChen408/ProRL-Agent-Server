import { useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTasks } from "../api/queries";
import { StatusPill } from "../components/StatusPill";
import { RewardChart } from "../components/RewardChart";
import { relativeTime, shortId, formatMs, formatReward } from "../utils";

const STATUS_OPTIONS = ["all", "running", "completed", "failed"];

export function TasksList() {
  const navigate = useNavigate();
  const { data, isLoading } = useTasks(2000);
  const [statusFilter, setStatusFilter] = useState("all");
  const [harnessFilter, setHarnessFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const tasks = data?.tasks ?? [];
  const harnessOptions = useMemo(() => {
    const set = new Set<string>();
    tasks.forEach((t) => t.harness && set.add(t.harness));
    return ["all", ...Array.from(set).sort()];
  }, [tasks]);

  const filtered = useMemo(() => {
    return tasks.filter((t) => {
      if (statusFilter !== "all" && t.status !== statusFilter) return false;
      if (harnessFilter !== "all" && t.harness !== harnessFilter) return false;
      if (search && !t.task_id.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [tasks, statusFilter, harnessFilter, search]);

  const toggleCompare = (taskId: string) => {
    setCompareIds((prev) =>
      prev.includes(taskId)
        ? prev.filter((id) => id !== taskId)
        : prev.length < 2
          ? [...prev, taskId]
          : [prev[1], taskId],
    );
  };

  const rewards = filtered.map((t) => t.mean_reward ?? null);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h1 className="mb-2 text-lg font-semibold">Tasks</h1>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-xs text-slate-500">Status</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {STATUS_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-500">Harness</label>
            <select
              value={harnessFilter}
              onChange={(e) => setHarnessFilter(e.target.value)}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {harnessOptions.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1 min-w-48">
            <label className="block text-xs text-slate-500">Search task id</label>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="substring"
              className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
            />
          </div>
          {compareIds.length === 2 && (
            <button
              type="button"
              className="rounded bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-700"
              onClick={() =>
                navigate(`/compare?a=${compareIds[0]}&b=${compareIds[1]}`)
              }
            >
              Compare 2 selected →
            </button>
          )}
        </div>
      </div>

      <RewardChart rewards={rewards} />

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
        {isLoading && (
          <div className="px-4 py-8 text-center text-sm text-slate-500">Loading…</div>
        )}
        {!isLoading && filtered.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-slate-500">
            No tasks match the filters.
          </div>
        )}
        {filtered.length > 0 && (
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2">cmp</th>
                <th className="px-3 py-2">task_id</th>
                <th className="px-3 py-2">status</th>
                <th className="px-3 py-2">harness</th>
                <th className="px-3 py-2">model</th>
                <th className="px-3 py-2">reward</th>
                <th className="px-3 py-2">sessions</th>
                <th
                  className="px-3 py-2"
                  title="Wall-clock execution time for single-session tasks"
                >
                  session_time
                </th>
                <th className="px-3 py-2">updated</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((task) => (
                <tr
                  key={task.task_id}
                  className="border-t border-slate-100 hover:bg-slate-50"
                >
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={compareIds.includes(task.task_id)}
                      onChange={() => toggleCompare(task.task_id)}
                    />
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link
                      to={`/tasks/${task.task_id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {shortId(task.task_id, 50)}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <StatusPill status={task.status} />
                  </td>
                  <td className="px-3 py-2 text-xs">{task.harness ?? "—"}</td>
                  <td className="px-3 py-2 text-xs">{task.model ?? "—"}</td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {formatReward(task.mean_reward)}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {task.completed_sessions}/{task.num_samples}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {formatMs(task.session_time_ms)}
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
