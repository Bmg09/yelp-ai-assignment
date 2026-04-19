import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def class_dist(rows: list[dict], key: str = "stars") -> dict[int, int]:
    d: dict[int, int] = {}
    for r in rows:
        d[r[key]] = d.get(r[key], 0) + 1
    return dict(sorted(d.items()))
