import type { SessionArtifactsPayload } from "../api/types";

interface Props {
  payload: SessionArtifactsPayload | undefined;
  loading?: boolean;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MiB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

export function ArtifactList({ payload, loading }: Props) {
  if (loading) return <div className="text-sm text-slate-500">Loading artifacts…</div>;
  const artifacts = payload?.artifacts ?? [];
  if (!artifacts.length) {
    return <div className="text-sm text-slate-500">No persisted profiling artifacts.</div>;
  }
  return (
    <div className="space-y-3">
      <div className="text-xs text-slate-500">
        {artifacts.length} files · {formatBytes(payload?.total_bytes ?? 0)}
        {(payload?.skipped_bytes ?? 0) > 0 && ` · ${formatBytes(payload?.skipped_bytes ?? 0)} skipped by budget`}
      </div>
      <div className="overflow-x-auto rounded border border-slate-200">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50 text-slate-500">
            <tr><th className="px-3 py-2">kind</th><th>path</th><th>size</th><th>sha256</th><th /></tr>
          </thead>
          <tbody>
            {artifacts.map((artifact) => (
              <tr key={artifact.id} className="border-t border-slate-100">
                <td className="px-3 py-2">{artifact.kind}</td>
                <td className="font-mono">{artifact.relative_path}</td>
                <td>{formatBytes(artifact.size_bytes)}</td>
                <td className="font-mono" title={artifact.sha256}>{artifact.sha256.slice(0, 12)}…</td>
                <td className="px-3 text-right"><a className="text-blue-600 hover:underline" href={artifact.download_url}>download</a></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
