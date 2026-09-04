import { useEffect, useMemo, useState } from "react";
import type { ChromeTraceEvent, SessionTracePayload } from "../api/types";
import { JsonView } from "./JsonView";

interface Props {
  trace: SessionTracePayload | undefined;
  loading?: boolean;
  unavailable?: boolean;
  onOpenCompletion?: (turn: number) => void;
  onOpenTrajectory?: (turn: number) => void;
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

export function TraceWaterfall({
  trace,
  loading,
  unavailable,
  onOpenCompletion,
  onOpenTrajectory,
}: Props) {
  const [query, setQuery] = useState("");
  const [processId, setProcessId] = useState("all");
  const [selected, setSelected] = useState<ChromeTraceEvent | null>(null);
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null);
  const spans = useMemo(() => spanEvents(trace?.trace_events ?? []), [trace]);
  const fullStart = spans.length ? Number(spans[0].ts) : 0;
  const fullEnd = Math.max(
    ...spans.map((span) => Number(span.ts) + Number(span.dur)),
    fullStart + 1,
  );
  const start = zoomRange?.[0] ?? fullStart;
  const end = zoomRange?.[1] ?? fullEnd;
  const extent = Math.max(1, end - start);

  useEffect(() => {
    setSelected(null);
    setZoomRange(null);
  }, [trace?.session_id]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return spans.filter((span) => {
      const processMatches = processId === "all" || String(span.pid) === processId;
      const textMatches = !needle || `${span.name ?? ""} ${span.cat ?? ""}`.toLowerCase().includes(needle);
      const spanStart = Number(span.ts);
      const spanEnd = spanStart + Number(span.dur);
      const windowMatches = spanEnd >= start && spanStart <= end;
      return processMatches && textMatches && windowMatches;
    });
  }, [spans, processId, query, start, end]);

  if (loading) return <div className="text-sm text-slate-500">Loading trace…</div>;
  if (unavailable || !trace) {
    return <div className="text-sm text-slate-500">No persisted trace is available for this session.</div>;
  }

  const selectedTurn = eventTurn(selected);
  const zoomToSelected = () => {
    if (!selected) return;
    const selectedStart = Number(selected.ts);
    const selectedDuration = Math.max(1, Number(selected.dur));
    const padding = Math.max(selectedDuration * 2, (fullEnd - fullStart) * 0.0025);
    setZoomRange([
      Math.max(fullStart, selectedStart - padding),
      Math.min(fullEnd, selectedStart + selectedDuration + padding),
    ]);
  };

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
        {zoomRange && (
          <button type="button" className="rounded border border-slate-300 px-2 py-1" onClick={() => setZoomRange(null)}>
            Reset zoom
          </button>
        )}
        <a className="ml-auto text-blue-600 hover:underline" href={trace.download_url}>Download Perfetto JSON</a>
      </div>
      <div className="max-h-[560px] overflow-auto rounded border border-slate-200">
        {filtered.map((span, index) => {
          const visibleStart = Math.max(start, Number(span.ts));
          const visibleEnd = Math.min(end, Number(span.ts) + Number(span.dur));
          const left = ((visibleStart - start) / extent) * 100;
          const width = Math.max(0.25, ((visibleEnd - visibleStart) / extent) * 100);
          const offsetMs = (Number(span.ts) - fullStart) / 1000;
          const durationMs = Number(span.dur) / 1000;
          return (
            <button
              type="button"
              key={`${span.pid}-${span.name}-${span.ts}-${index}`}
              onClick={() => setSelected(span)}
              className={`grid w-full grid-cols-[240px_1fr] border-b border-slate-100 text-left text-[11px] last:border-b-0 hover:bg-blue-50 ${selected === span ? "bg-blue-50" : ""}`}
            >
              <div className="truncate px-2 py-1 font-mono" title={span.name}>{span.name ?? "unnamed"}</div>
              <div className="relative my-1 mr-2 h-5 rounded bg-slate-50">
                <div
                  className={`absolute h-5 rounded ${PROCESS_COLORS[Number(span.pid)] ?? "bg-slate-400"}`}
                  style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
                  title={`${PROCESS_NAMES[Number(span.pid)] ?? span.pid} · +${offsetMs.toFixed(2)} ms · ${durationMs.toFixed(2)} ms`}
                />
              </div>
            </button>
          );
        })}
        {!filtered.length && <div className="p-4 text-sm text-slate-500">No spans match the filter.</div>}
      </div>
      {selected && (
        <div className="rounded border border-blue-200 bg-blue-50 p-3 text-xs">
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <span className="font-mono font-medium">{selected.name ?? "unnamed"}</span>
            <span>process: {PROCESS_NAMES[Number(selected.pid)] ?? selected.pid}</span>
            <span>offset: {((Number(selected.ts) - fullStart) / 1000).toFixed(2)} ms</span>
            <span>duration: {(Number(selected.dur) / 1000).toFixed(2)} ms</span>
            <button type="button" className="rounded border border-blue-500 px-2 py-1 text-blue-700" onClick={zoomToSelected}>
              Zoom to event
            </button>
            {selectedTurn != null && (
              <>
                <button type="button" className="ml-auto rounded bg-blue-600 px-2 py-1 text-white" onClick={() => onOpenCompletion?.(selectedTurn)}>
                  Open completion {selectedTurn}
                </button>
                <button type="button" className="rounded border border-blue-500 px-2 py-1 text-blue-700" onClick={() => onOpenTrajectory?.(selectedTurn)}>
                  Open trajectory
                </button>
              </>
            )}
          </div>
          <JsonView value={selected.args ?? {}} collapsed={false} maxHeight="260px" />
        </div>
      )}
    </div>
  );
}

function eventTurn(event: ChromeTraceEvent | null): number | null {
  if (!event) return null;
  const round = event.args?.round;
  if (typeof round === "number" && Number.isInteger(round) && round > 0) return round;
  const match = event.name?.match(/llm_call_(\d+)/);
  return match ? Number(match[1]) : null;
}
