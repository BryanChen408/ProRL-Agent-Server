import { statusClass } from "../utils";

interface Props {
  status?: string | null;
  className?: string;
}

export function StatusPill({ status, className = "" }: Props) {
  const label = status || "unknown";
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusClass(
        status,
      )} ${className}`}
    >
      {label}
    </span>
  );
}
