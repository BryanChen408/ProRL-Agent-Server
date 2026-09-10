#!/usr/bin/env python3
"""Exclude reviewed no-compute tasks from the audited DRKernel v3 dataset.

Uses the installed pyarrow package. Writes a new dataset; never executes reference
code, modifies the source, or reloads a running trainer.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


# ponytail: scoped to the audited v3 bytes; review a new revision before extending this list.
SOURCE_SHA256 = "4ab9c64c204abfdf95d3f3e304c6aee20b01171d2427bd76e59647ff4671d9fd"
EXCLUDED_ROWS = {
    93, 115, 129, 202, 231, 301, 478, 746, 957, 963, 1033, 1046,
    1098, 1137, 1347, 1434, 1609, 1718, 1733, 1734, 1822, 1952, 1976,
}


def filter_dataset(source: Path, assets: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source differs from audited DRKernel v3; refusing index-based filtering")

    table = pq.read_table(source)
    rows = table.to_pylist()
    jsonl_path = assets / "operator_tasks.ascendc.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 2000 or len(lines) != len(rows):
        raise ValueError("Expected all 2000 original rows in the source and training JSONL")
    names = set()
    for index, (row, line) in enumerate(zip(rows, lines, strict=True)):
        record = json.loads(line)
        metadata = record["metadata"]
        name = record["label"]
        if (
            not isinstance(name, str) or not name or Path(name).name != name
            or name in {".", ".."} or name in names
            or metadata["op_name"] != name
            or metadata["source"]["row_index"] != index
            or metadata["source"]["uuid"] != row["extra_info"]["uuid"]
        ):
            raise ValueError(f"Training row identity mismatch at source row {index}")
        names.add(name)
        code = row["reward_model"]["ground_truth"].encode("utf-8")
        if (
            (assets / "op_tasks" / f"{name}.py").read_bytes() != code
            or metadata["task_source_sha256"] != hashlib.sha256(code).hexdigest()
        ):
            raise ValueError(f"Reference mismatch at source row {index}")

    keep = [i for i in range(len(rows)) if i not in EXCLUDED_ROWS]
    removed = [
        {"source_row_index": i, "uuid": rows[i]["extra_info"]["uuid"],
         "reason": "reviewed forward only returns input or metadata/views; no compute kernel"}
        for i in sorted(EXCLUDED_ROWS)
    ]
    output.mkdir(parents=True)
    parquet_path = output / "training.parquet"
    pq.write_table(table.take(keep), parquet_path)
    training_path = output / "operator_tasks.ascendc.jsonl"
    training_path.write_text("\n".join(lines[i] for i in keep) + "\n", encoding="utf-8")
    # Reuse the exact reference assets; JSONL keeps original IDs and source row indices.
    (output / "op_tasks").symlink_to((assets / "op_tasks").resolve(), target_is_directory=True)

    # Runnable full-data regression check: filtering must not rewrite retained samples.
    assert pq.read_table(parquet_path).to_pylist() == [rows[i] for i in keep]
    assert training_path.read_text(encoding="utf-8").splitlines() == [lines[i] for i in keep]
    assert len(keep) == 1977 and 478 not in keep
    manifest = {
        "source_parquet": str(source.resolve()), "source_sha256": SOURCE_SHA256,
        "source_training_jsonl": str(jsonl_path.resolve()),
        "source_training_jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        "input_rows": len(rows), "output_rows": len(keep), "excluded": removed,
        "difficulty_counts": dict(Counter(rows[i]["extra_info"]["difficulty_level"] for i in keep)),
        "retained_source_row_indices": keep,
        "validation": "PASS: all retained parquet fields and training JSONL rows unchanged",
        "scope": "Only 23 reviewed no-compute tasks excluded; RNG tasks and duplicates retained. "
                 "Other tasks have not been certified by NPU preflight. op_tasks symlinks source assets.",
    }
    (output / "filter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = filter_dataset(args.source, args.assets, args.output)
    print(f"{result['input_rows']} -> {result['output_rows']} tasks; {result['validation']}")
    print(f"Training JSONL: {args.output / 'operator_tasks.ascendc.jsonl'}")
