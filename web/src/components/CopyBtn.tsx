import { useState } from "react";
import { copyToClipboard } from "../utils";

interface Props {
  value: string;
  label?: string;
}

export function CopyBtn({ value, label = "copy" }: Props) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="ml-1 text-xs text-slate-500 hover:text-slate-900 underline"
      onClick={async () => {
        await copyToClipboard(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      title={value}
    >
      {copied ? "copied" : label}
    </button>
  );
}
