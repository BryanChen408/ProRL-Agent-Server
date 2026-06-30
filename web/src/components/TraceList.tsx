import { useEffect, useMemo, useState } from "react";
import type { Trace } from "../api/types";
import { JsonView } from "./JsonView";

interface Props {
  traces: Trace[];
}

// Role → tailwind background/border classes. Keeps tool_calls and structured
// content visible by rendering each message verbatim as JSON.
const ROLE_STYLES: Record<string, { border: string; bg: string; label: string }> = {
  system: { border: "border-amber-200", bg: "bg-amber-50", label: "text-amber-700" },
  user: { border: "border-slate-200", bg: "bg-slate-50", label: "text-slate-700" },
  assistant: { border: "border-blue-200", bg: "bg-blue-50", label: "text-blue-800" },
  tool: { border: "border-emerald-200", bg: "bg-emerald-50", label: "text-emerald-800" },
  function: { border: "border-emerald-200", bg: "bg-emerald-50", label: "text-emerald-800" },
};

const FALLBACK_STYLE = {
  border: "border-violet-200",
  bg: "bg-violet-50",
  label: "text-violet-800",
};

type Origin = "prompt" | "response";

function charCount(value: unknown): number {
  try {
    return JSON.stringify(value).length;
  } catch {
    return 0;
  }
}

function messageText(message: any): string {
  const content = message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => (typeof item?.text === "string" ? item.text : ""))
      .filter(Boolean)
      .join("");
  }
  return "";
}

function toolNames(message: any): string[] {
  return (message?.tool_calls ?? [])
    .map((call: any) => call?.function?.name ?? call?.name)
    .filter((name: any) => typeof name === "string" && name.length > 0);
}

function MessageBlock({
  message,
  origin,
  index,
}: {
  message: any;
  origin: Origin;
  index: number;
}) {
  const role = typeof message?.role === "string" ? message.role : "unknown";
  const style = ROLE_STYLES[role] ?? FALLBACK_STYLE;
  const chars = useMemo(() => charCount(message), [message]);
  const text = messageText(message);
  const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content : "";
  const tools = toolNames(message);
  const isEmptyAssistant =
    role === "assistant" && !text && !reasoning && tools.length === 0;
  return (
    <div className={`rounded-lg border ${style.border} ${style.bg} p-2`}>
      <div className={`mb-1 flex items-center justify-between text-[10px] font-medium uppercase tracking-wide ${style.label}`}>
        <span>{role}</span>
        <span className="font-normal normal-case text-slate-500">
          #{index + 1} · ({origin} message) · {chars.toLocaleString()} chars
        </span>
      </div>
      {(text || reasoning || tools.length > 0 || isEmptyAssistant) && (
        <div className="mb-2 space-y-1 rounded border border-white/70 bg-white/60 p-2 text-xs text-slate-700">
          {isEmptyAssistant && (
            <div className="font-medium text-rose-700">empty assistant message</div>
          )}
          {text && <pre className="whitespace-pre-wrap break-words">{text}</pre>}
          {reasoning && (
            <details>
              <summary className="cursor-pointer font-medium text-slate-600">
                reasoning ({reasoning.length.toLocaleString()} chars)
              </summary>
              <pre className="mt-1 whitespace-pre-wrap break-words">{reasoning}</pre>
            </details>
          )}
          {tools.length > 0 && (
            <div className="font-medium text-slate-600">
              tool_calls: {tools.join(", ")}
            </div>
          )}
        </div>
      )}
      <JsonView value={message} collapsed={false} maxHeight="280px" />
    </div>
  );
}

function MessageGroup({
  title,
  origin,
  messages,
}: {
  title: string;
  origin: Origin;
  messages: any[];
}) {
  const totalChars = useMemo(
    () => messages.reduce((sum, m) => sum + charCount(m), 0),
    [messages],
  );
  return (
    <section className="space-y-2">
      <header className="flex items-center gap-3 border-b border-slate-200 pb-1">
        <h3 className="text-sm font-medium text-slate-700">{title}</h3>
        <span className="text-xs text-slate-500">
          {messages.length} message{messages.length === 1 ? "" : "s"} ·{" "}
          {totalChars.toLocaleString()} chars
        </span>
      </header>
      {messages.length === 0 ? (
        <div className="text-xs text-slate-500">(empty)</div>
      ) : (
        messages.map((m, mi) => (
          <MessageBlock key={mi} message={m} origin={origin} index={mi} />
        ))
      )}
    </section>
  );
}

function TraceView({ trace }: { trace: Trace }) {
  const prompt = trace.prompt_messages ?? [];
  const response = trace.response_messages ?? [];
  return (
    <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <div className="text-xs text-slate-500">
          reward {trace.reward != null ? trace.reward.toFixed(2) : "—"}
        </div>
        <div className="text-xs text-slate-500">
          finish_reason: {trace.finish_reason ?? "—"}
        </div>
        <div className="text-xs text-slate-500">
          {trace.response_ids?.length ?? 0} response tokens
        </div>
      </div>

      <MessageGroup title="Prompt messages" origin="prompt" messages={prompt} />
      <MessageGroup title="Response messages" origin="response" messages={response} />

      {trace.metadata && Object.keys(trace.metadata).length > 0 && (
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-slate-500">
            trace metadata
          </div>
          <JsonView value={trace.metadata} collapsed maxHeight="200px" />
        </div>
      )}
    </div>
  );
}

export function TraceList({ traces }: Props) {
  const [active, setActive] = useState(0);

  // Clamp active index when the trace count changes (e.g. while a session runs).
  useEffect(() => {
    if (active >= traces.length) {
      setActive(Math.max(0, traces.length - 1));
    }
  }, [traces.length, active]);

  if (!traces || traces.length === 0) {
    return <div className="text-sm text-slate-500">No traces reconstructed.</div>;
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1 overflow-x-auto rounded-lg border border-slate-200 bg-white p-1">
        {traces.map((trace, i) => {
          const isActive = i === active;
          const reward = trace.reward;
          const rewardLabel = reward == null ? "—" : reward.toFixed(2);
          return (
            <button
              key={i}
              type="button"
              onClick={() => setActive(i)}
              className={`whitespace-nowrap rounded px-3 py-1 text-xs ${
                isActive
                  ? "bg-blue-600 text-white"
                  : "text-slate-700 hover:bg-slate-100"
              }`}
              title={`reward ${rewardLabel} · finish ${trace.finish_reason ?? "—"}`}
            >
              Trace {i + 1}
              <span className={`ml-1 ${isActive ? "text-white/80" : "text-slate-500"}`}>
                ({rewardLabel})
              </span>
            </button>
          );
        })}
      </div>
      <TraceView trace={traces[active] ?? traces[0]} />
    </div>
  );
}
