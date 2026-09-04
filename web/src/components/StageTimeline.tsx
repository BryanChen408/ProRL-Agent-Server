import { formatMs } from "../utils";
import type { SessionTiming } from "../api/types";

interface Props {
  timing: SessionTiming | undefined;
}

const STAGE_ORDER: { key: keyof SessionTiming; label: string; color: string }[] = [
  { key: "register_to_init_queue_ms", label: "queue", color: "bg-slate-300" },
  { key: "init_ms", label: "init", color: "bg-violet-500" },
  { key: "run_ms", label: "run", color: "bg-blue-500" },
  { key: "postrun_ms", label: "postrun", color: "bg-emerald-500" },
];

export function StageTimeline({ timing }: Props) {
  if (!timing) {
    return <div className="text-sm text-slate-500">No timing data.</div>;
  }
  const total =
    STAGE_ORDER.reduce(
      (sum, s) => sum + Math.max(0, Number(timing[s.key] || 0)),
      0,
    ) || 1;

  return (
    <div className="space-y-3">
      <div className="flex h-6 w-full overflow-hidden rounded border border-slate-200 bg-slate-100">
        {STAGE_ORDER.map((stage) => {
          const v = Math.max(0, Number(timing[stage.key] || 0));
          const pct = (v / total) * 100;
          if (pct <= 0) return null;
          return (
            <div
              key={stage.key}
              className={`${stage.color} flex items-center justify-center text-[10px] text-white`}
              style={{ width: `${pct}%` }}
              title={`${stage.label}: ${formatMs(v)}`}
            >
              {pct > 8 ? stage.label : ""}
            </div>
          );
        })}
      </div>
      <table className="w-full text-xs">
        <tbody>
          {STAGE_ORDER.map((stage) => (
            <tr key={stage.key} className="border-b last:border-b-0">
              <td className="py-1 text-slate-500">{stage.label}</td>
              <td className="py-1 text-right font-mono">
                {formatMs(Number(timing[stage.key] || 0))}
              </td>
            </tr>
          ))}
          <tr className="font-medium">
            <td className="pt-2">total</td>
            <td className="pt-2 text-right font-mono">{formatMs(total)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
