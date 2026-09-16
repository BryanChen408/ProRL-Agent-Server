export function shortId(value?: string | null, maxLength = 16): string {
  if (!value) return "—";
  return value.length > maxLength
    ? `${value.slice(0, Math.max(0, maxLength - 1))}…`
    : value;
}

export function formatReward(value?: number | null): string {
  return value != null && Number.isFinite(value) ? value.toFixed(3) : "—";
}

export function formatMs(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 1000) return `${Math.round(value)} ms`;
  if (Math.abs(value) < 60000) return `${(value / 1000).toFixed(1)} s`;
  return `${(value / 60000).toFixed(1)} min`;
}

// API timestamps are Unix seconds.
export function relativeTime(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const seconds = Math.max(0, (Date.now() - value * 1000) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function statusClass(status?: string | null): string {
  switch (status?.toLowerCase()) {
    case "running":
      return "bg-blue-100 text-blue-700";
    case "completed":
    case "success":
      return "bg-green-100 text-green-700";
    case "error":
    case "failed":
      return "bg-red-100 text-red-700";
    case "timeout":
      return "bg-amber-100 text-amber-700";
    case "build":
    case "building":
      return "bg-purple-100 text-purple-700";
    default:
      return "bg-slate-100 text-slate-600";
  }
}

export async function copyToClipboard(value: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(value);
    return;
  }

  // Support dashboards served over HTTP on a remote host.
  const previousFocus = document.activeElement;
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  try {
    textarea.select();
    if (!document.execCommand("copy")) {
      throw new Error("Could not copy to clipboard");
    }
  } finally {
    textarea.remove();
    if (previousFocus instanceof HTMLElement) previousFocus.focus();
  }
}
