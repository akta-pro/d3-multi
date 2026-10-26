"""Score multi-label evaluation sets against a /v1/systemone server, three ways.

- multi: one `choice` question with the cutoff option; keep options that outscore it.
- noul: one yes/no question per label in one request, which the server runs as one pass per label.
- choice: the same `choice` question without the cutoff; keep options above a probability threshold.

python -m train.evaluate run --url http://localhost:8100 --model d3-mini --data data_out --out runs/base
python -m train.evaluate report --run runs/base
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from pathlib import Path

import httpx

from d3multi import multi

MODES = ("multi", "noul", "choice")


def noul_instructions(row: dict, key: str) -> str:
    desc = row["criteria"][key]
    label = f"{key} ({desc})" if desc else key
    return f"{row['instructions']}\nDoes this option apply: {label}?"


MAX_QUESTIONS = 64


def requests(row: dict, mode: str, model: str) -> list[dict]:
    if mode != "noul":
        return [request(row, mode, model)]
    keys = list(row["criteria"])
    return [{"model": model, "state": row["state"],
             "questions": {f"n{i}": {"type": "noul", "instructions": noul_instructions(row, keys[i])}
                           for i in range(start, min(start + MAX_QUESTIONS, len(keys)))}}
            for start in range(0, len(keys), MAX_QUESTIONS)]


def request(row: dict, mode: str, model: str) -> dict:
    if mode == "multi":
        q = {"q": multi.to_choice({"type": "multi", "instructions": row["instructions"], "criteria": row["criteria"]})}
    elif mode == "choice":
        q = {"q": {"type": "choice", "instructions": row["instructions"], "criteria": row["criteria"]}}
    else:
        q = {f"n{i}": {"type": "noul", "instructions": noul_instructions(row, k)} for i, k in enumerate(row["criteria"])}
    return {"model": model, "state": row["state"], "questions": q}


def scores(row: dict, mode: str, answers: dict) -> dict:
    if mode == "noul":
        return {k: answers[f"n{i}"]["noul"] for i, k in enumerate(row["criteria"])}
    return answers["q"]["probabilities"]


async def run(url: str, model: str, data: Path, out: Path, modes, limit: int, concurrency: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        for path in sorted(data.glob("eval_*.jsonl")):
            rows = [json.loads(l) for l in open(path)][:limit]
            for mode in modes:
                dest = out / f"{path.stem}.{mode}.jsonl"
                done = {json.loads(l)["id"] for l in open(dest)} if dest.exists() else set()
                todo = [r for r in rows if r["id"] not in done]

                async def one(r):
                    async with sem:
                        for attempt in range(4):
                            try:
                                answers = {}
                                for body in requests(r, mode, model):
                                    resp = await client.post(f"{url}/v1/systemone", json=body)
                                    resp.raise_for_status()
                                    answers.update(resp.json()["answers"])
                                return r, scores(r, mode, answers)
                            except (httpx.HTTPError, KeyError) as e:
                                err = e
                                await asyncio.sleep(2 ** attempt)
                        raise RuntimeError(f"{r['id']} {mode}: {err}")

                with open(dest, "a") as f:
                    for fut in asyncio.as_completed([one(r) for r in todo]):
                        r, s = await fut
                        f.write(json.dumps({"id": r["id"], "gold": r["gold"], "keys": list(r["criteria"]), "scores": s}) + "\n")
                print(f"{path.stem} {mode}: {len(todo)} new, {len(done)} kept", flush=True)


def f1_counts(pred: set, gold: set) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def metrics(rows: list[dict], select) -> dict:
    tp = fp = fn = 0
    sample_f1 = exact = 0.0
    per_label = {}
    for r in rows:
        pred, gold = set(select(r)), set(r["gold"])
        a, b, c = f1_counts(pred, gold)
        tp, fp, fn = tp + a, fp + b, fn + c
        sample_f1 += 1.0 if not pred and not gold else 2 * a / max(1, 2 * a + b + c)
        exact += pred == gold
        for k in r["keys"]:
            s = per_label.setdefault(k, [0, 0, 0])
            s[0] += k in pred and k in gold; s[1] += k in pred and k not in gold; s[2] += k in gold and k not in pred
    macro = [2 * a / (2 * a + b + c) for a, b, c in per_label.values() if a + b + c]
    n = max(1, len(rows))
    return {"micro_f1": 2 * tp / max(1, 2 * tp + fp + fn), "macro_f1": sum(macro) / max(1, len(macro)),
            "sample_f1": sample_f1 / n, "exact": exact / n, "labels_per_row": (tp + fp) / n}


def cv_threshold(rows: list[dict], key_fn, grid, seed=0) -> tuple[float, dict]:
    """Two-fold cross-validated threshold: each half is scored with the value tuned on the other half."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    halves = [[rows[i] for i in idx[: len(idx) // 2]], [rows[i] for i in idx[len(idx) // 2:]]]
    picked, chosen = [], {}
    for tune, test in (halves, halves[::-1]):
        best = max(grid, key=lambda t: metrics(tune, lambda r: key_fn(r, t))["micro_f1"])
        picked.append(best)
        chosen.update({r["id"]: key_fn(r, best) for r in test})
    m = metrics(rows, lambda r: chosen[r["id"]])
    return sum(picked) / 2, m


def ece(rows: list[dict], prob_fn, bins=10, floor=0.0) -> float:
    pts = [(p, k in r["gold"]) for r in rows for k in r["keys"] if (p := prob_fn(r, k)) >= floor]
    total, err = len(pts), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pts if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if sel:
            err += abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel)) * len(sel)
    return err / max(1, total)


def report(run_dir: Path) -> dict:
    out = {}
    stems = sorted({p.name.split(".")[0] for p in run_dir.glob("eval_*.jsonl")})
    offsets = [x / 4 for x in range(-16, 25)]
    probs = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for stem in stems:
        res = {}
        for mode in MODES:
            p = run_dir / f"{stem}.{mode}.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(l) for l in open(p)]
            if not rows:
                continue
            res["n"] = max(res.get("n", 0), len(rows))
            if mode == "multi":
                res["multi@0"] = metrics(rows, lambda r: multi.decode(r["scores"])["selected"])
                c, m = cv_threshold(rows, lambda r, t: multi.decode(r["scores"], t)["selected"], offsets)
                res["multi@cv"] = {**m, "offset": c}
                res["multi@0"]["ece"] = ece(rows, lambda r, k: multi.decode(r["scores"])["scores"][k])
                res["multi@0"]["ece_active"] = ece(rows, lambda r, k: multi.decode(r["scores"])["scores"][k], floor=0.05)
            elif mode == "noul":
                res["noul@0.5"] = metrics(rows, lambda r: [k for k, v in r["scores"].items() if v > 0.5])
                c, m = cv_threshold(rows, lambda r, t: [k for k, v in r["scores"].items() if v > t], [x / 20 for x in range(1, 20)])
                res["noul@cv"] = {**m, "threshold": c}
                res["noul@0.5"]["ece"] = ece(rows, lambda r, k: r["scores"][k])
                res["noul@0.5"]["ece_active"] = ece(rows, lambda r, k: r["scores"][k], floor=0.05)
            else:
                res["choice@top1"] = metrics(rows, lambda r: [max(r["scores"], key=r["scores"].get)])
                c, m = cv_threshold(rows, lambda r, t: [k for k, v in r["scores"].items() if v > t], probs)
                res["choice@cv"] = {**m, "threshold": c}
        out[stem] = res
    json.dump(out, open(run_dir / "report.json", "w"), indent=1)
    for stem, res in out.items():
        print(f"\n{stem} (n={res['n']})")
        for k, v in res.items():
            if k != "n":
                extra = {x: round(v[x], 4) for x in ("offset", "threshold", "ece", "ece_active") if x in v}
                print(f"  {k:12} micro {v['micro_f1']:.3f} macro {v['macro_f1']:.3f} sample {v['sample_f1']:.3f} "
                      f"exact {v['exact']:.3f} labels/row {v['labels_per_row']:.2f} {extra}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--data", default="data_out")
    r.add_argument("--out", required=True)
    r.add_argument("--modes", nargs="*", default=list(MODES))
    r.add_argument("--limit", type=int, default=10**9)
    r.add_argument("--concurrency", type=int, default=32)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a.url, a.model, Path(a.data), Path(a.out), a.modes, a.limit, a.concurrency))
    else:
        report(Path(a.run))


if __name__ == "__main__":
    main()
