"""A trainable d3-mini: the frozen backbone with LoRA on the language model, scored through the readout.

The forward pass matches the served model: non-causal attention on the full-attention layers, the
recurrent mask on the linear-attention layers, last-token pooling, then the 255-row readout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from d3multi import d3_format

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")
MAX_TOKENS = 8192


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        device = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))

    def forward(self, x):
        y = self.base(x)
        delta = (self.drop(x).to(torch.bfloat16) @ self.A.to(torch.bfloat16).T) @ self.B.to(torch.bfloat16).T
        return y + delta * self.scale


class D3Trainable:
    def __init__(self, root, device="cuda:0", r=16, alpha=32, dropout=0.05, lora=True):
        self.root = Path(root)
        self.device = torch.device(device)
        config = json.load(open(self.root / "decision_config.json"))
        assert config["attention_mode"] == "noncausal_full_attention" and config.get("pooling", "last") == "last"
        self.temperature = float(config.get("temperature", 1.0))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.root))
        self.codes = d3_format.answer_codes(self.tokenizer)[0]
        self.backbone = Qwen3_5Model.from_pretrained(str(self.root), dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to(self.device)
        self.backbone.requires_grad_(False)
        self.lm = self.backbone.language_model
        weight = load_file(str(self.root / "readout.safetensors"))["weight"]
        self.readout = nn.Parameter(weight.to(self.device, torch.float32))
        self.lora_modules = {}
        if lora:
            for name, module in list(self.lm.named_modules()):
                for child_name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear) and child_name in LORA_TARGETS:
                        wrapped = LoRALinear(child, r, alpha, dropout)
                        setattr(module, child_name, wrapped)
                        self.lora_modules[f"{name}.{child_name}" if name else child_name] = wrapped
        depth = self.lm.config.num_hidden_layers
        self.stack = [(layer, kind == "full_attention")
                      for layer, kind in zip(self.lm.layers[:depth], self.lm.config.layer_types[:depth])]

    def trainable_parameters(self):
        return [p for m in self.lora_modules.values() for p in (m.A, m.B)], [self.readout]

    def render_ids(self, state, question):
        text = d3_format.render(self.tokenizer, state, question, self.codes)
        return self.tokenizer([text], add_special_tokens=False)["input_ids"][0]

    def logits(self, ids, n_options, grad_checkpoint=True):
        ids_t = torch.as_tensor([ids], dtype=torch.long, device=self.device)
        embeds = self.backbone.get_input_embeddings()(ids_t)
        width = ids_t.shape[1]
        ramp = torch.arange(width, device=self.device).view(1, 1, -1).expand(4, 1, -1)
        text_positions, mrope = ramp[0], ramp[1:]
        rope = self.lm.rotary_emb(embeds, mrope)
        present = torch.ones((1, width), dtype=torch.long, device=self.device)
        keys = present.bool()[:, None, None, :]
        scan = create_recurrent_attention_mask(config=self.lm.config, inputs_embeds=embeds, attention_mask=present,
                                               past_key_values=None, position_ids=text_positions)
        hidden = embeds
        for layer, softmax in self.stack:
            def run(h, layer=layer, softmax=softmax):
                return layer(h, position_embeddings=rope, attention_mask=keys if softmax else scan,
                             position_ids=text_positions, past_key_values=None, use_cache=False)
            if grad_checkpoint and torch.is_grad_enabled():
                hidden = checkpoint(run, hidden, use_reentrant=False)
            else:
                hidden = run(hidden)
        last = self.lm.norm(hidden)[:, -1]
        return (last.float() @ self.readout.T)[0, :n_options] / self.temperature

    def lora_state(self):
        out = {}
        for name, m in self.lora_modules.items():
            out[f"{name}.lora_A"] = m.A.detach().cpu().contiguous()
            out[f"{name}.lora_B"] = m.B.detach().cpu().contiguous()
        out["readout.weight"] = self.readout.detach().cpu().contiguous()
        return out

    def load_lora_state(self, state):
        for name, m in self.lora_modules.items():
            m.A.data.copy_(state[f"{name}.lora_A"].to(m.A.device))
            m.B.data.copy_(state[f"{name}.lora_B"].to(m.B.device))
        self.readout.data.copy_(state["readout.weight"].to(self.readout.device))
