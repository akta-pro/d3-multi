"""Fine-tune d3-mini for the `multi` question type while keeping its `choice`, `noul` and `score` answers.

python -m train.train --base ~/d3-mini-base --multi data_out/train.jsonl --distill data_out/distill.jsonl --out ckpt

Two losses share every step:
- multi: binary cross-entropy on each option's margin over the cutoff, z_i - z_REF, against its gold label;
- distill: KL divergence from the base model's own option probabilities on choice, noul and score questions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from d3multi import multi
from train.model import MAX_TOKENS, D3Trainable


def multi_tensors(model, row, rng):
    keys = list(row["criteria"])
    if rng is not None:
        rng.shuffle(keys)
    question = multi.to_choice({"instructions": row["instructions"], "criteria": {k: row["criteria"][k] for k in keys}})
    order = list(question["criteria"])
    if rng is not None and rng.random() < 0.5:
        rng.shuffle(order)
        question["criteria"] = {k: question["criteria"][k] for k in order}
    if len(order) > len(model.codes):
        return None
    ids = model.render_ids(row["state"], question)
    if len(ids) > MAX_TOKENS:
        return None
    gold = set(row["gold"])
    ref = order.index(multi.REF)
    target = torch.tensor([float(k in gold) for k in order if k != multi.REF])
    return ids, len(order), ref, target


def distill_tensors(model, row):
    question = row["question"]
    n = 2 if question["type"] == "noul" else len(question["criteria"])
    ids = model.render_ids(row["state"], question)
    if len(ids) > MAX_TOKENS:
        return None
    probs = torch.tensor(row["probs"], dtype=torch.float32).clamp_min(1e-8)
    return ids, n, probs / probs.sum()


def margins(logits, ref):
    keep = [i for i in range(logits.shape[0]) if i != ref]
    return logits[keep] - logits[ref]


def multi_loss(model, item, grad=True):
    ids, n, ref, target = item
    m = margins(model.logits(ids, n, grad_checkpoint=grad), ref)
    return F.binary_cross_entropy_with_logits(m, target.to(m.device)), m


def distill_loss(model, item, grad=True):
    ids, n, probs = item
    logp = F.log_softmax(model.logits(ids, n, grad_checkpoint=grad), dim=-1)
    probs = probs.to(logp.device)
    return (probs * (probs.log() - logp)).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--multi", required=True)
    ap.add_argument("--distill")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--readout-lr", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--distill-weight", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-multi", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    model = D3Trainable(a.base, r=a.rank, alpha=2 * a.rank)
    lora_params, readout_params = model.trainable_parameters()
    groups = [{"params": lora_params, "lr": a.lr, "weight_decay": 0.0, "base_lr": a.lr}]
    if a.readout_lr > 0:
        groups.append({"params": readout_params, "lr": a.readout_lr, "weight_decay": 0.01, "base_lr": a.readout_lr})
    else:
        model.readout.requires_grad_(False)
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in groups for p in g["params"]]

    multi_rows = [json.loads(l) for l in open(a.multi)]
    random.Random(a.seed).shuffle(multi_rows)
    val_multi, multi_rows = multi_rows[: a.val_multi], multi_rows[a.val_multi:]
    distill_rows = [json.loads(l) for l in open(a.distill)] if a.distill else []
    random.Random(a.seed).shuffle(distill_rows)
    val_distill, distill_rows = distill_rows[:300], distill_rows[300:]
    tasks = [("multi", r) for r in multi_rows] + [("distill", r) for r in distill_rows]
    total_steps = max(1, int(len(tasks) // a.accum * a.epochs))
    warmup_steps = max(1, int(total_steps * a.warmup))
    print(f"tasks {len(tasks)} (distill {len(distill_rows)}), val multi {len(val_multi)} distill {len(val_distill)}, "
          f"steps {total_steps}", flush=True)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))))

    def evaluate():
        model.backbone.eval()
        tp = fp = fn = 0
        mloss = dloss = 0.0
        nm = nd = 0
        with torch.no_grad():
            for r in val_multi:
                item = multi_tensors(model, r, None)
                if item is None:
                    continue
                loss, m = multi_loss(model, item, grad=False)
                pred, gold = (m > 0).cpu(), item[3] >= 0.5
                tp += int((pred & gold).sum()); fp += int((pred & ~gold).sum()); fn += int((~pred & gold).sum())
                mloss += float(loss); nm += 1
            for r in val_distill:
                item = distill_tensors(model, r)
                if item is not None:
                    dloss += float(distill_loss(model, item, grad=False)); nd += 1
        model.backbone.train()
        return {"val_multi_loss": mloss / max(1, nm), "val_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "val_distill_kl": dloss / max(1, nd), "val_n": nm}

    def save(tag):
        save_file(model.lora_state(), str(out / f"lora_{tag}.safetensors"))
        torch.save(opt.state_dict(), out / f"optim_{tag}.pt")
        json.dump({"step": step, "seen": seen, "tag": tag}, open(out / "latest.tmp", "w"))
        os.replace(out / "latest.tmp", out / "latest.json")
        for p in sorted(out.glob("lora_step*.safetensors"), key=lambda p: p.stat().st_mtime)[:-2]:
            p.unlink(missing_ok=True)
            (out / p.name.replace("lora_", "optim_").replace(".safetensors", ".pt")).unlink(missing_ok=True)

    step, seen = 0, 0
    if (out / "latest.json").exists() and not a.smoke:
        meta = json.load(open(out / "latest.json"))
        model.load_lora_state(load_file(str(out / f"lora_{meta['tag']}.safetensors")))
        opt.load_state_dict(torch.load(out / f"optim_{meta['tag']}.pt", weights_only=False))
        step, seen = meta["step"], meta["seen"]
        print(f"resumed at step {step}", flush=True)

    rng = random.Random(a.seed + step)
    if a.smoke:
        longest = max((t for t in tasks if t[0] == "multi"), key=lambda t: len(t[1]["state"]) + 40 * len(t[1]["criteria"]))[1]
        for kind, item in (("multi", multi_tensors(model, longest, rng)), ("distill", distill_tensors(model, distill_rows[0]) if distill_rows else None)):
            if item is None:
                continue
            t0 = time.time()
            loss = multi_loss(model, item)[0] if kind == "multi" else distill_loss(model, item)
            loss.backward()
            print(f"smoke {kind}: {len(item[0])} tokens, {item[1]} options, loss {float(loss):.4f}, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, {time.time() - t0:.2f}s", flush=True)
        raise SystemExit(0)

    if step == 0:
        v = evaluate()
        log.write(json.dumps({"step": 0, **v}) + "\n"); log.flush()
        print("step 0", v, flush=True)

    wall0 = t0 = last = time.time()
    model.backbone.train()
    run = {"multi": [0.0, 0], "distill": [0.0, 0]}
    micro = skipped = 0
    epoch = seen // len(tasks)
    while step < total_steps:
        idx = list(range(len(tasks)))
        random.Random(a.seed + epoch).shuffle(idx)
        for i in idx[seen - epoch * len(tasks):]:
            seen += 1
            kind, row = tasks[i]
            item = multi_tensors(model, row, rng) if kind == "multi" else distill_tensors(model, row)
            if item is None:
                skipped += 1
                continue
            loss = multi_loss(model, item)[0] if kind == "multi" else a.distill_weight * distill_loss(model, item)
            (loss / a.accum).backward()
            run[kind][0] += float(loss); run[kind][1] += 1
            micro += 1
            if micro % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if time.time() - last > 120:
                rate = micro / (time.time() - t0)
                rec = {"step": step, "multi_loss": run["multi"][0] / max(1, run["multi"][1]),
                       "distill_kl": run["distill"][0] / max(1, run["distill"][1]), "ex_per_s": rate, "skipped": skipped}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"step {step}/{total_steps} multi {rec['multi_loss']:.4f} kl {rec['distill_kl']:.4f} {rate:.2f} ex/s "
                      f"eta {(total_steps - step) * a.accum / max(rate, 1e-6) / 3600:.1f}h", flush=True)
                run = {"multi": [0.0, 0], "distill": [0.0, 0]}
                micro = 0
                t0 = last = time.time()
            if step % a.ckpt_every == 0:
                save(f"step{step}")
            if step % a.val_every == 0:
                v = evaluate()
                log.write(json.dumps({"step": step, **v}) + "\n"); log.flush()
                print(f"step {step}", v, flush=True)
            if step >= total_steps or (time.time() - wall0) / 3600 > a.max_hours:
                break
        else:
            epoch += 1
            continue
        break

    save(f"step{step}")
    save("final")
    v = evaluate()
    log.write(json.dumps({"step": step, **v, "final": True}) + "\n"); log.flush()
    print("DONE", step, v, flush=True)


if __name__ == "__main__":
    main()
