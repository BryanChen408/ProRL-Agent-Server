"""Compare saved developer systems with a pinned cannbot source, without installing it.

Exit 2 means the historical workflow is not matched; do not use it as permission
to replace missing skills with similarly named current ones. Counts are files,
not independent benchmark trials or measured success rates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def developer_body(text):
    marker = "# Ascend Kernel Developer"
    return text[text.index(marker):].strip() if marker in text else ""


def summarize(samples, source_text):
    body = developer_body(source_text)
    counts = Counter()
    variants = {}
    matched = 0
    for path in sorted(samples.rglob("*__sub_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        system = "\n".join(
            msg["content"] for msg in data.get("messages", [])
            if msg.get("role") == "system" and isinstance(msg.get("content"), str)
        )
        observed = developer_body(system)
        if not observed:
            continue
        digest = hashlib.sha256(observed.encode()).hexdigest()
        counts[digest] += 1
        exact = bool(body) and observed == body
        matched += int(exact)
        outline = re.search(r"## 工作流总览\s*```([^`]+)```", observed)
        variants.setdefault(digest, {
            "body_sha256": digest,
            "example": str(path.relative_to(samples)),
            "exact_source_match": exact,
            "outline": outline.group(1).strip() if outline else None,
        })
    return {
        "sample_files": sum(counts.values()),
        "exact_source_matches": matched,
        "status": "matched" if matched and matched == sum(counts.values()) else "unverified",
        "variants": [{**variants[digest], "files": n} for digest, n in counts.most_common()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--agent-path", required=True, help="Agent markdown path inside the pinned git tree")
    parser.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    args = parser.parse_args()
    if not args.samples.is_dir():
        parser.error("samples must be an existing directory")

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(args.repo), *argv], text=True)

    commit = git("rev-parse", "--verify", args.ref + "^{commit}").strip()
    source = git("show", f"{commit}:{args.agent_path}")
    report = {"source_commit": commit, "agent_path": args.agent_path,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              **summarize(args.samples, source)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(encoded)
        print(f"{report['status']}: {report['exact_source_matches']}/{report['sample_files']} files; {args.output}")
    else:
        print(encoded, end="")
    return 0 if report["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
