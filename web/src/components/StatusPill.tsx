import type { ReactNode } from "react";
import { statusClass } from "../lib/utils";

interface Props {
  status?: string | null;
  className?: string;
  children?: ReactNode;
}

export function StatusPill({ status, className = "", children }: Props) {
  const label = status || "unknown";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusClass(
        status,
      )} ${className}`}
    >
      {children ?? label}
    </span>
  );
}
