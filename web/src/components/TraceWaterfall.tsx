import { useMemo, useState } from "react";
import type { ChromeTraceEvent, SessionTracePayload } from "../api/types";

interface Props {
  trace: SessionTracePayload | undefined;
  loading?: boolean;
  unavailable?: boolean;
}

const PROCESS_NAMES: Record<number, string> = { 1: "Gateway", 2: "Engine", 3: "Agent" };
const PROCESS_COLORS: Record<number, string> = {
  1: "bg-blue-500",
  2: "bg-fuchsia-500",
  3: "bg-emerald-500",
};

function spanEvents(events: ChromeTraceEvent[]): ChromeTraceEvent[] {
  return events
    .filter((event) => event.ph === "X" && Number.isFinite(event.ts) && Number.isFinite(event.dur))
    .sort((left, right) => Number(left.ts) - Number(right.ts));
}

export function TraceWaterfall({ trace, loading, unavailable }: Props) {
  const [query, setQuery] = useState("");
  const [processId, setProcessId] = useState("all");
  const spans = useMemo(() => spanEvents(trace?.trace_events ?? []), [trace]);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return spans.filter((span) => {
      const processMatches = processId === "all" || String(span.pid) === processId;
      const textMatches = !needle || `${span.name ?? ""} ${span.cat ?? ""}`.toLowerCase().includes(needle);
      return processMatches && textMatches;
    });
  }, [spans, processId, query]);

  if (loading) return <div className="text-sm text-slate-500">Loading trace…</div>;
  if (unavailable || !trace) {
    return <div className="text-sm text-slate-500">No persisted trace is available for this session.</div>;
  }

  const start = spans.length ? Number(spans[0].ts) : 0;
  const end = Math.max(...spans.map((span) => Number(span.ts) + Number(span.dur)), start + 1);
  const extent = Math.max(1, end - start);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <input
          className="min-w-64 rounded border border-slate-300 px-2 py-1"
          placeholder="Filter span name or category"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={processId}
          onChange={(event) => setProcessId(event.target.value)}
        >
          <option value="all">All processes</option>
          {[1, 2, 3].map((pid) => <option key={pid} value={pid}>{PROCESS_NAMES[pid]}</option>)}
        </select>
        <span className="text-slate-500">schema v{trace.schema_version} · {filtered.length}/{spans.length} spans</span>
        <a className="ml-auto text-blue-600 hover:underline" href={trace.download_url}>Download Perfetto JSON</a>
      </div>
      <div className="max-h-[560px] overflow-auto rounded border border-slate-200">
        {filtered.map((span, index) => {
          const left = ((Number(span.ts) - start) / extent) * 100;
          const width = Math.max(0.25, (Number(span.dur) / extent) * 100);
          const offsetMs = (Number(span.ts) - start) / 1000;
          const durationMs = Number(span.dur) / 1000;
          return (
            <div key={`${span.pid}-${span.name}-${span.ts}-${index}`} className="grid grid-cols-[240px_1fr] border-b border-slate-100 text-[11px] last:border-b-0">
              <div className="truncate px-2 py-1 font-mono" title={span.name}>{span.name ?? "unnamed"}</div>
              <div className="relative my-1 mr-2 h-5 rounded bg-slate-50">
                <div
                  className={`absolute h-5 rounded ${PROCESS_COLORS[Number(span.pid)] ?? "bg-slate-400"}`}
                  style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
                  title={`${PROCESS_NAMES[Number(span.pid)] ?? span.pid} · +${offsetMs.toFixed(2)} ms · ${durationMs.toFixed(2)} ms`}
                />
              </div>
            </div>
          );
        })}
        {!filtered.length && <div className="p-4 text-sm text-slate-500">No spans match the filter.</div>}
      </div>
    </div>
  );
}
