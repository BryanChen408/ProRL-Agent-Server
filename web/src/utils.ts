export function shortId(value: unknown, limit = 16): string {
  const text = value == null ? "" : String(value);
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 1))}…`;
}

export function formatMs(value: unknown): string {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";
  if (amount < 1) return `${amount.toFixed(2)} ms`;
  if (amount < 1_000) return `${amount.toFixed(0)} ms`;
  if (amount < 60_000) return `${(amount / 1_000).toFixed(2)} s`;
  return `${(amount / 60_000).toFixed(2)} min`;
}

export function formatReward(value: unknown): string {
  const amount = Number(value);
  return Number.isFinite(amount) ? amount.toFixed(4) : "—";
}

export function relativeTime(value: unknown): string {
  const raw = Number(value);
  if (!Number.isFinite(raw) || raw <= 0) return "—";
  const timestampMs = raw < 10_000_000_000 ? raw * 1_000 : raw;
  const deltaSeconds = Math.round((Date.now() - timestampMs) / 1_000);
  const future = deltaSeconds < 0;
  const absolute = Math.abs(deltaSeconds);
  let text: string;
  if (absolute < 60) text = `${absolute}s`;
  else if (absolute < 3_600) text = `${Math.round(absolute / 60)}m`;
  else if (absolute < 86_400) text = `${Math.round(absolute / 3_600)}h`;
  else text = `${Math.round(absolute / 86_400)}d`;
  return future ? `in ${text}` : `${text} ago`;
}

export function statusClass(status: string | null | undefined): string {
  switch ((status ?? "").toUpperCase()) {
    case "COMPLETED":
    case "READY":
      return "bg-emerald-100 text-emerald-800";
    case "ERROR":
    case "FAILED":
      return "bg-red-100 text-red-800";
    case "TIMEOUT":
      return "bg-amber-100 text-amber-800";
    case "RUNNING":
    case "INITIALIZING":
    case "POST_RUN":
      return "bg-blue-100 text-blue-800";
    default:
      return "bg-slate-100 text-slate-700";
  }
}

export async function copyToClipboard(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}
