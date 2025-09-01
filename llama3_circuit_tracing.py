#!/usr/bin/env python3

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# 1) CLI and setup
# -----------------------------

def parse_layers_heads(arg: str, max_inclusive: int) -> List[int]:
	"""Parse comma-separated indices or 'all' into a list of ints within [0, max_inclusive]."""
	arg = arg.strip().lower()
	if arg == "all":
		return list(range(max_inclusive + 1))
	if not arg:
		return []
	out: List[int] = []
	for part in arg.split(","):
		part = part.strip()
		if not part:
			continue
		idx = int(part)
		if idx < 0 or idx > max_inclusive:
			raise ValueError(f"Index {idx} out of range [0, {max_inclusive}]")
		out.append(idx)
	return sorted(set(out))


def setup_logging(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)  # Path.mkdir per spec
	log_path = output_dir / "run.log"
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(message)s",
		handlers=[
			logging.StreamHandler(),
			logging.FileHandler(log_path, mode="w", encoding="utf-8"),
		],
	)


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Llama-3-8B-Instruct circuit tracing with head ablations.")  # argparse
	p.add_argument("--output_dir", type=str, default="outputs/llama3_circuit_tracing")
	p.add_argument("--num_samples", type=int, default=3)
	p.add_argument("--max_new_tokens", type=int, default=0)  # not used (no generation), kept for completeness
	p.add_argument("--layers", type=str, default="all")
	p.add_argument("--heads", type=str, default="all")
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--bf16", action="store_true")
	p.add_argument("--device_map", type=str, default="auto")
	p.add_argument("--topk", type=int, default=20, help="Top-K absolute mean effects to show in the plot")
	return p


# -----------------------------
# 2) Model load and chat templating helpers
# -----------------------------

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # model card / Llama3 docs


def load_model_and_tokenizer(bf16: bool, device_map: str):
	# torch.cuda.is_available to decide dtype preference; device_map="auto" per spec
	dtype = torch.bfloat16 if bf16 else None
	logging.info(f"Loading tokenizer: {MODEL_ID}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)  # AutoTokenizer
	logging.info(f"Loading model: {MODEL_ID} (bf16={bf16}, device_map={device_map})")
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		torch_dtype=dtype,
		device_map=device_map,
	)
	return model, tokenizer


def apply_chat(tokenizer, messages: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
	# apply_chat_template add_generation_prompt=True per spec
	input_ids = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_tensors="pt",
	)
	return input_ids.to(device)


def eos_terminators(tokenizer) -> List[int]:
	# eos and eot handling as per model card examples
	eos = []
	if tokenizer.eos_token_id is not None:
		eos.append(tokenizer.eos_token_id)
	try:
		eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
		if eot_id is not None and eot_id != tokenizer.unk_token_id:
			eos.append(eot_id)
	except Exception:
		pass
	return eos


# -----------------------------
# 3) Suppression task and delta-logit metric
# -----------------------------

TARGET_CANDIDATES = [
	"Paris", "London", "Einstein", "football", "banana", "apple", "music", "science", "ocean",
]


def pick_targets_for_sample(rng: np.random.Generator, k: int = 3) -> List[str]:
	k = max(1, min(k, len(TARGET_CANDIDATES)))
	choices = rng.choice(TARGET_CANDIDATES, size=k, replace=False)
	return list(choices)


def to_single_token_ids(tokenizer, tokens: Sequence[str]) -> List[Tuple[str, Optional[int]]]:
	# Ensure each target corresponds to a single next-token id when preceded by space.
	ids: List[Tuple[str, Optional[int]]] = []
	for t in tokens:
		# prepend space for correct BPE alignment (Tokenizer docs)
		enc = tokenizer.encode(" " + t, add_special_tokens=False)
		if len(enc) == 1:
			ids.append((t, enc[0]))
		else:
			logging.warning(f"Skipping multi-token target '{t}' (ids={enc}) for next-token metric")
			ids.append((t, None))
	return ids


def neutral_and_suppress_prompts(targets: Sequence[str]) -> Tuple[str, str, int]:
	# Compose simple neutral vs suppression prompts consistent with spec
	# neutral sentence embeds generic context; suppression adds constraints not to mention targets
	neutral = (
		"Answer concisely about general knowledge. Provide a brief statement."
	)
	suppress_list = ", ".join(targets)
	suppress = f"Don't mention {suppress_list}."
	return neutral, suppress, len(targets)


def messages_from_prompts(neutral: str, suppress: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
	msgs_neutral = [
		{"role": "system", "content": "You are an assistant."},
		{"role": "user", "content": neutral},
	]
	msgs_suppress = [
		{"role": "system", "content": "You are an assistant."},
		{"role": "user", "content": suppress},
	]
	return msgs_neutral, msgs_suppress


def compute_log_probs_last(model, input_ids: torch.Tensor) -> torch.Tensor:
	# Forward pass without sampling; obtain logits at last position; softmax to probs, then log
	outputs = model(input_ids, use_cache=False)  # CausalLMOutputWithPast
	logits = outputs.logits  # (bs, seq, vocab)
	last_logits = logits[:, -1, :]
	log_probs = torch.log_softmax(last_logits, dim=-1)  # torch.nn.functional.softmax -> log
	return log_probs


def delta_logit_for_targets(model, tokenizer, msgs_neutral, msgs_suppress, device: torch.device, target_ids: List[Tuple[str, Optional[int]]]) -> float:
	with torch.inference_mode():  # inference_mode per spec
		ids_neutral = apply_chat(tokenizer, msgs_neutral, device)
		ids_suppress = apply_chat(tokenizer, msgs_suppress, device)
		logp_neutral = compute_log_probs_last(model, ids_neutral)
		logp_suppress = compute_log_probs_last(model, ids_suppress)
		diffs: List[float] = []
		for _, tid in target_ids:
			if tid is None:
				continue
			base_ln = logp_neutral[0, tid].item()
			sup_ln = logp_suppress[0, tid].item()
			diffs.append(sup_ln - base_ln)
		if not diffs:
			return float("nan")
		return float(np.mean(diffs))  # averaged over targets


# -----------------------------
# 4) Head ablation hooks on o_proj
# -----------------------------

class HeadAblator:
	"""Register forward_pre_hook on each layer's self_attn.o_proj to zero specific head slices.

	The hook reads an active_ablations dict: {layer_idx: set(head_indices)}
	and zeroes hidden_states[..., head_start:head_end] for each head.
	"""

	def __init__(self, model) -> None:
		self.model = model
		self.handles: List[torch.utils.hooks.RemovableHandle] = []
		self.active: Dict[int, Set[int]] = {}
		self.hidden_size: int = int(model.config.hidden_size)
		self.num_heads: int = int(model.config.num_attention_heads)
		self.head_dim: int = self.hidden_size // self.num_heads
		self.layer_count: int = len(model.model.layers)

		# Map layer index -> module for o_proj
		self.layer_to_module: Dict[int, torch.nn.Module] = {}
		for name, module in model.named_modules():  # named_modules reference
			if name.endswith("self_attn.o_proj"):
				# Extract layer index from path like 'model.layers.12.self_attn.o_proj'
				parts = name.split(".")
				try:
					idx = parts.index("layers")
					layer_idx = int(parts[idx + 1])
					self.layer_to_module[layer_idx] = module
				except Exception:
					continue

		# Register per-layer hook
		for layer_idx, module in self.layer_to_module.items():
			handle = module.register_forward_pre_hook(self._make_pre_hook(layer_idx))  # forward_pre_hook per spec
			self.handles.append(handle)

	def _make_pre_hook(self, layer_idx: int):
		def hook(module, inputs):
			# inputs is a tuple; first element is hidden_states [bs, seq, hidden]
			if not self.active or layer_idx not in self.active or len(self.active[layer_idx]) == 0:
				return None  # no-op
			try:
				hs = inputs[0]
			except Exception:
				return None
			if not torch.is_tensor(hs):
				return None
			# Create a copy to avoid in-place on shared tensor passed further
			hs = hs.clone()
			for h in self.active[layer_idx]:
				start = h * self.head_dim
				end = (h + 1) * self.head_dim
				# Zero the slice for all time steps; aligns heads via concatenation ordering
				hs[..., start:end] = 0
			# Return new inputs tuple
			return (hs, *inputs[1:])
		return hook

	def clear(self):
		self.active.clear()

	def set_single(self, layer: int, head: int):
		self.active = {layer: {head}}

	def close(self):
		for h in self.handles:
			try:
				h.remove()
			except Exception:
				pass
		self.handles.clear()


# -----------------------------
# 5) Main experiment flow
# -----------------------------

def main():
	args = build_argparser().parse_args()
	output_dir = Path(args.output_dir)
	setup_logging(output_dir)  # logging, Path

	# Determinism / seeds
	torch.manual_seed(args.seed)  # torch.manual_seed per spec
	np.random.seed(args.seed)

	# Load model/tokenizer
	model, tokenizer = load_model_and_tokenizer(args.bf16, args.device_map)
	device = model.device  # use model.device when device_map="auto"
	logging.info(f"Model device: {device}, hidden_size={model.config.hidden_size}, heads={model.config.num_attention_heads}")

	# Prepare layer/head selections
	n_layers = len(model.model.layers)
	n_heads = int(model.config.num_attention_heads)
	layers = parse_layers_heads(args.layers, n_layers - 1)
	heads = parse_layers_heads(args.heads, n_heads - 1)
	logging.info(f"Selected layers: {layers if layers else '[]'}; heads: {heads if heads else '[]'}")

	# Hook manager
	ablator = HeadAblator(model)

	rows: List[Dict[str, object]] = []
	rng = np.random.default_rng(args.seed)

	try:
		for sample_id in range(args.num_samples):
			target_tokens = pick_targets_for_sample(rng, k=3)
			target_ids = to_single_token_ids(tokenizer, target_tokens)
			neutral, suppress, n_constraints = neutral_and_suppress_prompts(target_tokens)
			msgs_neutral, msgs_suppress = messages_from_prompts(neutral, suppress)

			# Baseline delta_logit
			delta_base = delta_logit_for_targets(
				model, tokenizer, msgs_neutral, msgs_suppress, device, target_ids
			)
			logging.info(f"sample={sample_id} baseline delta_logit={delta_base:.6f} targets={target_tokens}")

			# Sweep heads
			for layer in layers:
				for head in heads:
					ablator.set_single(layer, head)
					with torch.inference_mode():
						# recompute under ablation for both prompts
						ids_neutral = apply_chat(tokenizer, msgs_neutral, device)
						ids_suppress = apply_chat(tokenizer, msgs_suppress, device)
						logp_neutral = compute_log_probs_last(model, ids_neutral)
						logp_suppress = compute_log_probs_last(model, ids_suppress)
						diffs: List[float] = []
						for _, tid in target_ids:
							if tid is None:
								continue
							base_ln = logp_neutral[0, tid].item()
							sup_ln = logp_suppress[0, tid].item()
							diffs.append(sup_ln - base_ln)
						delta_head = float(np.mean(diffs)) if diffs else float("nan")

						effect = delta_head - delta_base if (not math.isnan(delta_head) and not math.isnan(delta_base)) else float("nan")
						rows.append(
							{
								"sample_id": sample_id,
								"n_constraints": n_constraints,
								"layer": layer,
								"head": head,
								"delta_logit_base": delta_base,
								"delta_logit_head": delta_head,
								"head_effect": effect,
							}
						)
	finally:
		ablator.clear()

	# Save CSV
	df = pd.DataFrame(rows)
	csv_path = output_dir / "circuit_results.csv"
	df.to_csv(csv_path, index=False)  # pandas to_csv per spec
	logging.info(f"Wrote CSV: {csv_path} ({len(df)} rows)")

	# Simple bar plot of top-K absolute mean effects
	if len(df) > 0:
		agg = (
			df.groupby(["layer", "head"], as_index=False)["head_effect"].mean()
			.rename(columns={"head_effect": "mean_head_effect"})
		)
		# sort by abs and take top-K
		agg["sort_key"] = agg["mean_head_effect"].abs()
		agg = agg.sort_values("sort_key", ascending=False).head(max(1, args.topk))
		labels = [f"L{int(r.layer)}H{int(r.head)}" for r in agg.itertuples(index=False)]
		y = agg["mean_head_effect"].tolist()
		plt.figure(figsize=(max(6, len(labels) * 0.5), 4))
		plt.bar(range(len(labels)), y)
		plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
		plt.ylabel("mean_head_effect")
		plt.title("Top head effects (delta_head - delta_base)")
		plt.tight_layout()
		png_path = output_dir / "circuit_summary.png"
		plt.savefig(png_path, dpi=300)  # savefig per spec
		plt.close()
		logging.info(f"Wrote PNG: {png_path}")
	else:
		logging.error("No rows were produced; nothing to save.")
		raise SystemExit(1)


if __name__ == "__main__":
	main() 