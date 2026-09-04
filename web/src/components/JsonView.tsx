import { useMemo, useState } from "react";
import { copyToClipboard } from "../utils";

interface Props {
  value: any;
  collapsed?: boolean;
  maxHeight?: string;
}

export function JsonView({ value, collapsed = false, maxHeight = "320px" }: Props) {
  const [open, setOpen] = useState(!collapsed);
  const text = useMemo(() => {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }, [value]);

  if (!open) {
    return (
      <button
        type="button"
        className="text-xs text-slate-500 underline"
        onClick={() => setOpen(true)}
      >
        show JSON ({text.length.toLocaleString()} chars)
      </button>
    );
  }

  return (
    <div className="rounded border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-2 py-1 text-xs text-slate-600">
        <span>{text.length.toLocaleString()} chars</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="underline"
            onClick={() => copyToClipboard(text)}
          >
            copy
          </button>
          {collapsed && (
            <button
              type="button"
              className="underline"
              onClick={() => setOpen(false)}
            >
              collapse
            </button>
          )}
        </div>
      </div>
      <pre
        className="json overflow-auto p-2"
        style={{ maxHeight }}
      >
        {text}
      </pre>
    </div>
  );
}
