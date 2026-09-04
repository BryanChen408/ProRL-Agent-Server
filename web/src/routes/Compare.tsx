import { useSearchParams, Link } from "react-router-dom";
import { useTask } from "../api/queries";
import { RewardChart } from "../components/RewardChart";
import { StatusPill } from "../components/StatusPill";
import { formatMs, formatReward } from "../utils";

function summarize(data: any) {
  const sessions: any[] = data?.sessions ?? [];
  const rewards = sessions
    .map((s) => s.reward)
    .filter((r): r is number => typeof r === "number");
  const mean = rewards.length
    ? rewards.reduce((a, b) => a + b, 0) / rewards.length
    : null;
  const completed = sessions.filter((s) => s.status === "COMPLETED").length;
  const errored = sessions.filter((s) => ["ERROR", "TIMEOUT"].includes(s.status))
    .length;
  const init = sessions.map((s) => Number(s.timing?.init_ms || 0));
  const run = sessions.map((s) => Number(s.timing?.run_ms || 0));
  const postrun = sessions.map((s) => Number(s.timing?.postrun_ms || 0));
  const mean_of = (arr: number[]) =>
    arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  return {
    sessions,
    rewards: sessions.map((s) => s.reward ?? null),
    mean,
    completed,
    errored,
    total: sessions.length,
    init_mean: mean_of(init),
    run_mean: mean_of(run),
    postrun_mean: mean_of(postrun),
  };
}

export function Compare() {
  const [params] = useSearchParams();
  const a = params.get("a") ?? "";
  const b = params.get("b") ?? "";
  const aTask = useTask(a);
  const bTask = useTask(b);

  if (!a || !b) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        Provide ?a=task_id&amp;b=task_id, or use the Tasks page → select two checkboxes →
        Compare.
      </div>
    );
  }

  const A = aTask.data ? summarize(aTask.data) : null;
  const B = bTask.data ? summarize(bTask.data) : null;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h1 className="text-lg font-semibold">Compare</h1>
        <div className="mt-1 text-xs text-slate-500">
          <Link to={`/tasks/${a}`} className="text-blue-600 underline">
            {a}
          </Link>{" "}
          vs{" "}
          <Link to={`/tasks/${b}`} className="text-blue-600 underline">
            {b}
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[
          { label: "A", task: aTask.data, sum: A },
          { label: "B", task: bTask.data, sum: B },
        ].map(({ label, task, sum }) => (
          <div key={label} className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="font-mono text-sm">{task?.task_id ?? "—"}</div>
              <StatusPill status={task?.status} />
            </div>
            {sum && (
              <>
                <RewardChart rewards={sum.rewards} />
                <table className="mt-3 w-full text-xs">
                  <tbody>
                    <tr>
                      <td className="text-slate-500">Mean reward</td>
                      <td className="text-right font-mono">{formatReward(sum.mean)}</td>
                    </tr>
                    <tr>
                      <td className="text-slate-500">Completed</td>
                      <td className="text-right font-mono">
                        {sum.completed}/{sum.total}
                      </td>
                    </tr>
                    <tr>
                      <td className="text-slate-500">Errored</td>
                      <td className="text-right font-mono">{sum.errored}</td>
                    </tr>
                    <tr>
                      <td className="text-slate-500">Mean init</td>
                      <td className="text-right font-mono">
                        {formatMs(sum.init_mean)}
                      </td>
                    </tr>
                    <tr>
                      <td className="text-slate-500">Mean run</td>
                      <td className="text-right font-mono">{formatMs(sum.run_mean)}</td>
                    </tr>
                    <tr>
                      <td className="text-slate-500">Mean postrun</td>
                      <td className="text-right font-mono">
                        {formatMs(sum.postrun_mean)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
