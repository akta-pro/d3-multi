"""Merge a trained LoRA into the base d3-mini and write a BF16 model directory that serves as-is.

python -m train.merge --base ~/d3-mini-base --lora ckpt/lora_final.safetensors --rank 16 --out d3-mini-multi

Several --lora files with --weights average their weight deltas (a model soup): base + sum(w_k * delta_k).
One --lora with a weight below 1 blends the fine-tune toward the base (WiSE-FT): base + w * delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM_PREFIX = "model.language_model."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True, action="append")
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, required=True)
    a = ap.parse_args()

    base, out = Path(a.base), Path(a.out)
    scale = 2.0  # trained with alpha = 2 * rank
    weights = a.weights or [1.0 / len(a.lora)] * len(a.lora)
    assert len(weights) == len(a.lora), "one weight per --lora"
    loras = []
    for path in a.lora:
        with safe_open(path, "pt") as f:
            loras.append({k: f.get_tensor(k) for k in f.keys()})
    readout = sum(w * l.pop("readout.weight").float() for w, l in zip(weights, loras)) / sum(weights)
    mods = {k[: -len(".lora_A")] for k in loras[0] if k.endswith(".lora_A")}
    assert all({k[: -len(".lora_A")] for k in l if k.endswith(".lora_A")} == mods for l in loras), "LoRA files cover different modules"
    assert loras[0][next(iter(mods)) + ".lora_A"].shape[0] == a.rank, "rank does not match the LoRA file"

    def delta(mod):
        return sum(w * scale * (l[mod + ".lora_B"].float() @ l[mod + ".lora_A"].float()) for w, l in zip(weights, loras))

    index = json.load(open(base / "model.safetensors.index.json"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    merged, sums, weight_map = set(), {}, {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(str(base / shard), "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                mod = None
                if k.endswith(".weight"):
                    name = k[: -len(".weight")]
                    for prefix in (LM_PREFIX, "language_model.", "model."):
                        if name.startswith(prefix) and name[len(prefix):] in mods:
                            mod = name[len(prefix):]
                            break
                if mod is not None:
                    t = (t.float() + delta(mod)).to(t.dtype)
                    merged.add(mod)
                tensors[k] = t.contiguous()
                weight_map[k] = shard
                sums[k] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print("wrote", shard, flush=True)
    assert merged == mods, f"unmerged LoRA modules: {sorted(mods - merged)[:5]}"

    for f in base.iterdir():
        if f.is_file() and f.name not in {"MODEL_MANIFEST.json", "readout.safetensors"} and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    for d in base.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            shutil.copytree(d, out / d.name)
    json.dump({"metadata": index.get("metadata", {}), "weight_map": weight_map}, open(out / "model.safetensors.index.json", "w"), indent=2)
    save_file({"weight": readout.float().contiguous()}, str(out / "readout.safetensors"))
    json.dump(sums, open(out / "tensor_sha256.json", "w"), indent=0)
    print(f"BUILT {out}: merged {len(merged)} LoRA modules into {len(weight_map)} tensors, readout {tuple(readout.shape)}")


if __name__ == "__main__":
    main()
