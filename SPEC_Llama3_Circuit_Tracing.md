# Specification: Llama‑3‑8B‑Instruct Circuit Tracing Experiment (CSV + simple PNG) ([model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Objective: Implement a reproducible circuit tracing script that probes attention heads in `meta-llama/Meta-Llama-3-8B-Instruct`, measures head-level effects on a suppression-task delta-logit metric derived from your experiment, and writes a results `.csv` plus a single clean `.png` plot. ([model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3), [Causal LM outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- Constraints: Use official chat formatting for Llama-3-Instruct via `tokenizer.apply_chat_template`, use `AutoModelForCausalLM`/`AutoTokenizer`, collect per-layer/per-head measurements with PyTorch hooks, and keep the graph minimalistic. ([chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating), [Auto classes](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/auto), [PyTorch hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook))

- Deliverables: `llama3_circuit_tracing.py` script; `circuit_results.csv` with head effects; `circuit_summary.png` clean, simple plot; stdout log of key steps. ([pandas to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html), [matplotlib savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html))

## 1) Environment and Dependencies ([Transformers install](https://huggingface.co/docs/transformers/en/installation), [PyTorch](https://pytorch.org/get-started/locally/))

- Ensure Python 3.10+ and install: `pip install torch transformers accelerate pandas matplotlib tqdm`. ([Transformers install](https://huggingface.co/docs/transformers/en/installation), [PyTorch](https://pytorch.org/get-started/locally/))

- Optional: login to Hugging Face for gated model access: `huggingface-cli login`. ([Hub login CLI](https://huggingface.co/docs/hub/en/security-tokens), [model card access note](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct))

- Set deterministic seeds: `torch.manual_seed(42)` and `numpy.random.seed(42)` if used. ([torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html))

## 2) Script Skeleton and CLI ([argparse](https://docs.python.org/3/library/argparse.html), [logging](https://docs.python.org/3/library/logging.html), [pathlib.Path](https://docs.python.org/3/library/pathlib.html))

- Create `llama3_circuit_tracing.py` that accepts args: `--output_dir`, `--num_samples`, `--max_new_tokens`, `--layers`, `--heads`, `--seed`, `--bf16`, `--device_map`. ([argparse](https://docs.python.org/3/library/argparse.html))

- Initialize basic logging to stdout and file under `output_dir` using Python `logging`. ([logging](https://docs.python.org/3/library/logging.html))

- Create output directory with `Path(output_dir).mkdir(parents=True, exist_ok=True)`. ([pathlib.Path](https://docs.python.org/3/library/pathlib.html))

## 3) Model and Tokenizer Loading ([Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3), [Auto classes](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/auto))

- Use `model_id = "meta-llama/Meta-Llama-3-8B-Instruct"` to load tokenizer with `AutoTokenizer.from_pretrained(model_id)`. ([model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [AutoTokenizer](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/auto))

- Load model with `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if bf16 else None, device_map=device_map)`. ([Llama3 usage](https://huggingface.co/docs/transformers/en/model_doc/llama3), [device_map auto example](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Detect CUDA and set dtype preference: prefer BF16 if GPU supports; fallback to CPU offloading with `device_map="auto"` if memory constrained. ([torch.cuda.is_available](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html), [Llama3 device_map example](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Switch to inference mode: wrap generations/forwards in `with torch.inference_mode():` for performance. ([torch.inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html))

## 4) Prompt Formatting for Llama‑3‑Instruct ([chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating), [Llama3 prompt example](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Build chat messages per instruct format, e.g., `[{"role":"system","content":"You are an assistant."},{"role":"user","content": user_text}]`. ([chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating))

- Convert to input IDs via `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`, then move to the model device. ([apply_chat_template](https://huggingface.co/docs/transformers/main/en/chat_templating))

- Use `max_new_tokens` from CLI; pass `eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]` if available to stop correctly. ([Llama3 generation example](https://huggingface.co/docs/transformers/en/model_doc/llama3), [Tokenizers special tokens](https://huggingface.co/docs/transformers/en/main_classes/tokenizer))

## 5) Suppression Task (Delta‑Logit Metric) ([Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer), [Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- For each sample, construct a neutral sentence and its suppression-augmented counterpart (e.g., `"Don't mention A or B or C. " + sentence`) mirroring your experiment design. ([Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer))

- Run two forward passes to obtain logits for both prompts without sampling: call `model(input_ids, use_cache=False)` and read `outputs.logits`. ([Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- Compute per-target token probabilities at the aligned positions using `torch.nn.functional.softmax(logits[..., -1, :], dim=-1)` and extract token IDs for target words (prepend space for correct wordpiece alignment if needed). ([torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html), [Tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer))

- Define `delta_logit = log P_suppress(target) − log P_neutral(target)` averaged over listed targets, following the original experiment’s scoring. ([Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

## 6) Enumerating Layers and Heads ([Llama model](https://huggingface.co/docs/transformers/main/en/model_doc/llama), [named_modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules))

- Llama decoder layers live at `model.model.layers[i]` with submodules `self_attn` and `mlp`, where attention uses `num_attention_heads` and `hidden_size`. ([Llama model](https://huggingface.co/docs/transformers/main/en/model_doc/llama))

- Retrieve structural parameters from `model.config.hidden_size`, `model.config.num_attention_heads`, `head_dim = hidden_size // num_attention_heads`. ([Llama config](https://huggingface.co/docs/transformers/main/en/model_doc/llama))

- Iterate `for i, m in model.named_modules():` to locate attention output projections `*.self_attn.o_proj` in each layer for targeted head intervention points. ([named_modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules))

## 7) Head‑Level Intervention via Forward Pre‑Hooks ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook), [hooks guide](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook))

- Register a `forward_pre_hook` on each `self_attn.o_proj` to zero one or more head slices in its input tensor before projection, using head‑aligned slices `[..., h*head_dim:(h+1)*head_dim] = 0`. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook))

- The pre‑hook receives `(module, inputs)`; return a new `inputs` tuple with the masked tensor to apply ablation cleanly for a selected head. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook))

- Implement utilities to enable/disable hooks per `(layer_index, head_index)` so the script can sweep heads one at a time and measure `delta_logit` effects. ([hooks guide](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook))

## 8) Measurement Protocol ([inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html), [Causal LM outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- Baseline: compute `delta_logit` without ablation for each sample to get `delta_logit_base`. ([Causal LM outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- For each `(layer, head)` in selected sets, enable the pre‑hook to ablate that head for both neutral and suppression passes, and recompute `delta_logit_head`. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook))

- Define head effect as `delta_logit_head − delta_logit_base` (positive implies the head mitigates suppression; negative implies it contributes to suppression). ([Causal LM outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast))

- Run under `with torch.inference_mode():` to avoid autograd and maximize throughput. ([inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html))

## 9) Data Recording and CSV Output ([pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), [to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html))

- Accumulate rows: `sample_id, n_constraints, layer, head, delta_logit_base, delta_logit_head, head_effect`. ([pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html))

- Save to `circuit_results.csv` in `output_dir` using `DataFrame.to_csv(index=False)`. ([to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html))

## 10) Simple, Clean PNG Plot ([matplotlib pyplot](https://matplotlib.org/stable/api/pyplot_summary.html), [bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html), [savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html))

- Aggregate mean `head_effect` over samples per `(layer, head)` and select top‑K absolute effects for clarity. ([pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html))

- Create a minimal bar chart with x‑axis labeled `L{layer}H{head}` and y‑axis `mean_head_effect`, sorted by magnitude, no extra decoration beyond axis labels and title. ([bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html))

- Save plot to `circuit_summary.png` at 300 DPI. ([savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html))

## 11) Efficiency and Safety ([device_map auto](https://huggingface.co/docs/transformers/en/model_doc/llama3), [inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html))

- Default to `device_map="auto"` to enable CPU/GPU offloading when VRAM is limited; consider BF16 for supported GPUs. ([device_map example](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Ensure all hook registration/removal is exception‑safe (try/finally) to avoid leaving mutated modules in memory. ([hooks guide](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook))

- Run evaluations without gradient tracking to reduce memory and improve speed. ([inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html))

## 12) Reference Code Patterns ([Llama3 usage](https://huggingface.co/docs/transformers/en/model_doc/llama3), [chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating), [Auto classes](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/auto))

- Loading example for Llama‑3‑Instruct: see docs snippet using `AutoTokenizer`/`AutoModelForCausalLM`, BF16, and `device_map="auto"`. ([Llama3 usage](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- Chat template example with `add_generation_prompt=True` to ensure proper assistant continuation. ([apply_chat_template](https://huggingface.co/docs/transformers/main/en/chat_templating))

- General `Auto*` APIs: `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`. ([Auto classes](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/auto))

## 13) Optional: Output Attentions for Diagnostics ([output_attentions](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.output_attentions), [Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output))

- For sanity checks, you can pass `output_attentions=True` during forward to retrieve attention weights and confirm hook head targeting matches shapes `(batch, num_heads, seq, seq)`. ([output_attentions](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.output_attentions))

- Do not use attention weights for head masking; use the pre‑hook slice‑zeroing strategy to ablate exact head inputs into `o_proj`. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook))

## 14) End‑to‑End Flow Summary ([Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3), [PyTorch hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook), [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html), [matplotlib](https://matplotlib.org/stable/users/explain/figure/quick_start.html))

- Load tokenizer/model with BF16 if available and `device_map` from CLI; seed; set inference mode. ([Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3), [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html), [inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html))

- Build neutral vs suppression prompts per sample; format via chat template; get logits. ([chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating), [Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output))

- Compute base `delta_logit`; sweep `(layer, head)` with pre‑hook ablations; compute `head_effect`; store to DataFrame. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook), [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html))

- Write `circuit_results.csv`; render minimalist bar chart `circuit_summary.png`; exit with non‑zero on failures. ([to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html), [bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html), [savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html))

---

### Notes and Citations

- Llama‑3‑8B‑Instruct access and examples are on the model card and Transformers Llama3 page; ensure you have requested access where required. ([model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama3 docs](https://huggingface.co/docs/transformers/en/model_doc/llama3))

- `apply_chat_template` guarantees correct role formatting and generation prompt handling across chat models. ([chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating))

- PyTorch forward pre‑hooks allow modifying inputs before the module’s forward computation; use to zero specific head slices feeding `o_proj`. ([forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook))

- Use `torch.inference_mode()` (or `no_grad`) to avoid autograd overhead during evaluation. ([inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)) 