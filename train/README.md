# Training `multi`

This folder fine-tunes [d3-mini](https://huggingface.co/vllm-sr/d3-mini) to answer `multi` questions. It keeps
the model's `choice`, `noul` and `score` answers close to the base model's. Run every command from the
repository root.

You need one GPU with 32 GB of memory and a vLLM build that serves d3-mini.

## 1. Prepare the data

Bring two JSON Lines files.

**`train.jsonl`**: one `multi` question per line, with the options that apply.

```json
{"id": "1", "state": "The order arrived damaged and I was charged twice.", "instructions": "Which teams need to act on this request?", "criteria": {"returns": "Refunds, replacements and damaged deliveries", "billing": "Payments, invoices and charges", "shipping": null}, "gold": ["returns", "billing"]}
```

- `criteria` maps each option key to a description, or to `null` for none. Use at most 254 options.
- `gold` lists the keys that apply. An empty list means that no option applies.
- Include rows where no option applies, so the model learns to return an empty answer.

**`distill.jsonl`**: the base model's own answers to `choice` and `noul` questions. These keep the other question
types from changing while the model learns `multi`.

```json
{"state": "The order arrived damaged and I was charged twice.", "question": {"type": "noul", "instructions": "Was the customer charged twice?"}, "probs": [0.02, 0.98]}
{"state": "The order arrived damaged and I was charged twice.", "question": {"type": "choice", "instructions": "How urgent is this request?", "criteria": {"Routine": null, "Soon": null, "Today": null}}, "probs": [0.10, 0.35, 0.55]}
```

- Ask the base d3-mini each question and save the probabilities it returns.
- For a `choice` question, `probs` has one value per option, in the order of `criteria`.
- For a `noul` question, `probs` is `[no, yes]`.
- Send a `score` question as a `choice` question with its levels as the options.

## 2. Train

```bash
python -m train.train --base <d3-mini dir> --multi train.jsonl --distill distill.jsonl --out ckpt
```

The trainer adds a LoRA adapter to the language model and keeps the readout and vision tower frozen. Each step
combines two losses:

- **`multi`:** binary cross-entropy on each option's margin over the cutoff option.
- **Self-distillation:** the divergence from the base model's answers in `distill.jsonl`.

`--smoke` runs one step on the longest example and reports memory. The trainer saves a checkpoint every 100 steps
and resumes from the latest one.

## 3. Merge

```bash
python -m train.merge --base <d3-mini dir> --lora ckpt/lora_final.safetensors --rank 16 --out d3-mini-multi
```

The output is a BF16 model directory that vLLM serves as it is.

## 4. Evaluate

Put each test set in a folder as `eval_<name>.jsonl`, in the `train.jsonl` format.

```bash
vllm serve d3-mini-multi --served-model-name d3-mini-multi --port 8100 --no-enable-prefix-caching
python -m train.evaluate run --url http://localhost:8100 --model d3-mini-multi --data <eval folder> --out runs/multi
python -m train.evaluate report --run runs/multi
```

The report compares three ways to answer each test set:

- one `multi` question;
- one `noul` question per label;
- one `choice` question with a probability threshold.

These steps train a `multi` model. They do not reproduce the released weights exactly.
