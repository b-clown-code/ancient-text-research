"""Streaming causal-LM trainer for controlled Gate 3 optimizer comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .data import (WeightedRecordPool, iter_records, pack_records,
                   scheduled_records, sources_for_language, weighted_records)
from .model import AncientDecoder, ModelConfig, parameter_count
from .progressive import (estimate_training_flops, load_depth_schedule,
                          model_parameter_count, stage_for_tokens)


DEFAULT_WEIGHTS = {f"language/{lang}": 1.0 for lang in
                   ("grc", "lat", "san", "pli", "syr", "cop", "ara", "egy", "bod", "tam")}


def validate_curriculum(stages, languages):
    if not stages or stages[-1].get("end_fraction") != 1.0:
        raise ValueError("curriculum stages must end at token fraction 1.0")
    previous = 0.0
    expected = set(languages)
    for stage in stages:
        end = float(stage["end_fraction"])
        if not previous < end <= 1.0:
            raise ValueError("curriculum end fractions must strictly increase")
        previous = end
        totals = {lang: 0.0 for lang in expected}
        for spec, weight in stage["weights"].items():
            if spec.startswith("language/"):
                lang = spec.split("/", 1)[1]
            elif len(spec.split("/")) == 4 and spec.split("/")[2] == "language":
                lang = spec.split("/")[3]
            else:
                raise ValueError("curriculum keys must identify a language explicitly")
            if lang not in expected:
                raise ValueError(f"curriculum contains unsupported language {lang!r}")
            if weight < 0:
                raise ValueError("curriculum weights cannot be negative")
            totals[lang] += float(weight)
        if set(totals) != expected or any(abs(v - 1.0) > 1e-8 for v in totals.values()):
            raise ValueError(f"each language must retain weight 1.0 in stage {stage['name']}")


def curriculum_stage_index(stages, token_count, target_tokens):
    fraction = min(token_count / max(1, target_tokens), 1.0)
    for index, stage in enumerate(stages):
        if fraction < stage["end_fraction"] or index == len(stages) - 1:
            return index
    return len(stages) - 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def optimizer_state_bytes(optimizer):
    """Count tensor storage held in optimizer state."""
    total = 0
    for state in optimizer.state.values():
        total += sum(value.numel() * value.element_size()
                     for value in state.values() if torch.is_tensor(value))
    return total


def optimizer_factory(name, model, lr, weight_decay):
    if name != "adamw":
        raise ValueError("this curated trainer supports AdamW")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, tokenizer, device, languages, seq_len, batches_per_language):
    model.eval()
    results = {}
    for lang in languages:
        records = iter_records("validation", sources=sources_for_language(lang),
                               languages={lang})
        stream = pack_records(records, tokenizer, sequence_length=seq_len, batch_size=1)
        losses = []
        for i, (x, y) in enumerate(stream):
            if i >= batches_per_language:
                break
            losses.append(float(model(x.to(device), y.to(device))["loss"].float()))
        if losses:
            result = {"loss": sum(losses) / len(losses), "batches": len(losses)}
            if len(losses) >= 8:
                result["loss_first_8"] = sum(losses[:8]) / 8
            results[lang] = result
    model.train()
    return results


@torch.no_grad()
def generate_examples(model, tokenizer, device, languages, max_new_tokens=64):
    model.eval()
    samples = {}
    eos_id = tokenizer.token_to_id("<|eos|>")
    for lang in languages:
        try:
            row = next(iter_records("train", sources=sources_for_language(lang),
                                    languages={lang}))
        except StopIteration:
            continue
        prompt = row["text"][:120]
        ids = tokenizer.encode(prompt).ids[:64]
        if ids:
            generated = model.generate(torch.tensor([ids], dtype=torch.long, device=device),
                                       max_new_tokens, eos_token_id=eos_id)
            samples[lang] = {"prompt": prompt,
                             "generation": tokenizer.decode(generated[0].tolist())}
    model.train()
    return samples


def move_optimizer_state_to_device(optimizer, device):
    # Optimizer.load_state_dict does device casting in supported PyTorch builds;
    # make it explicit for the custom optimizer and ROCm/CUDA parity.
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)


def save_checkpoint(path, model, optimizer, config, seed, optimizer_name,
                    step, tokens, tokenizer_hash, weights, data_seed, init_checkpoint,
                    training_config, *, depth_state=None, validation_history=None,
                    elapsed_seconds=0.0):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "config": config, "seed": seed, "optimizer_name": optimizer_name,
               "step": step, "tokens": tokens, "tokenizer_sha256": tokenizer_hash,
               "sampling_weights": weights, "data_seed": data_seed,
               "init_checkpoint": init_checkpoint, "training_config": training_config,
               "depth_state": depth_state, "validation_history": validation_history or [],
               "elapsed_seconds": elapsed_seconds}
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def update_data_digest(digest, x, y):
    digest.update(x.contiguous().numpy().tobytes())
    digest.update(y.contiguous().numpy().tobytes())


def mean_validation_loss(by_language):
    losses = [item["loss"] for item in by_language.values()]
    return sum(losses) / len(losses) if losses else None


def optimizer_state_tensor_count(optimizer):
    return sum(1 for state in optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def set_stabilization_trainable(model, stage_index, stages, remaining):
    if remaining <= 0 or stage_index <= 0:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    start = int(stages[stage_index - 1]["n_layers"])
    end = int(stages[stage_index]["n_layers"])
    new_parameters = {id(parameter) for layer in model.layers[start:end]
                      for parameter in layer.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in new_parameters)


def fresh_stage_metrics(model, tokens, step):
    return {"n_layers": model.config.n_layers,
            "parameters": model_parameter_count(model.config),
            "start_tokens": tokens, "start_step": step,
            "tokens": 0, "optimizer_steps": 0,
            "training_compute_seconds": 0.0, "estimated_training_flops": 0,
            "peak_vram_allocated_bytes": 0,
            "peak_vram_reserved_bytes": 0}


def close_stage_metrics(stage, end_tokens, end_step, optimizer, device):
    record = dict(stage)
    record.update({"end_tokens": end_tokens, "end_step": end_step,
                   "tokens_per_second": (stage["tokens"] /
                                          max(stage["training_compute_seconds"], 1e-9)),
                   "optimizer_state_bytes_at_end": optimizer_state_bytes(optimizer)})
    if device.type == "cuda":
        record["peak_vram_allocated_bytes"] = max(
            record["peak_vram_allocated_bytes"],
            int(torch.cuda.max_memory_allocated(device)))
        record["peak_vram_reserved_bytes"] = max(
            record["peak_vram_reserved_bytes"],
            int(torch.cuda.max_memory_reserved(device)))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--sampling-mode", choices=("random", "curriculum"), default="random")
    ap.add_argument("--curriculum-config", type=Path,
                    help="External curriculum JSON; required for --sampling-mode curriculum")
    ap.add_argument("--tokens", type=int, default=1_000_000,
                    help="Total target tokens, including work before --resume")
    ap.add_argument("--sequence-length", type=int, choices=(512, 1024), default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--optimizer", choices=("adamw",), default="adamw")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--data-seed", type=int, default=20261102)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--stop-after-steps", type=int,
                    help="Stop at this global step after writing a resumable checkpoint")
    ap.add_argument("--init-checkpoint", type=Path,
                    help="Shared model-only initialization for paired optimizer runs")
    ap.add_argument("--depth-schedule", type=Path,
                    help="Optional fixed-stage progressive layer schedule JSON")
    ap.add_argument("--growth-stabilization-steps", type=int, default=3,
                    help="Train only newly added layers for this many steps after growth")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--skip-generation", action="store_true",
                    help="Omit diagnostic sample generation after training")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.sampling_mode == "curriculum":
        if args.curriculum_config is None:
            ap.error("--curriculum-config is required for curriculum sampling")
        if not args.curriculum_config.is_file():
            ap.error(f"curriculum config does not exist: {args.curriculum_config}")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_hash = sha256(args.tokenizer)
    weights = json.loads(args.weights.read_text()) if args.weights else DEFAULT_WEIGHTS
    languages = sorted({k.rsplit("/", 1)[-1] for k in weights if k.startswith("language/")})
    curriculum_data = None
    curriculum_stages = None
    if args.sampling_mode == "curriculum":
        curriculum_data = json.loads(args.curriculum_config.read_text())
        curriculum_stages = curriculum_data["stages"]
        validate_curriculum(curriculum_stages, languages)
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
                         max_seq_len=1024)
    progressive_schedule = (load_depth_schedule(args.depth_schedule,
                                                final_depth=config.n_layers,
                                                target_tokens=args.tokens)
                            if args.depth_schedule else None)
    if args.growth_stabilization_steps < 0:
        raise ValueError("growth stabilization steps cannot be negative")
    resume_checkpoint = (torch.load(args.resume, map_location="cpu", weights_only=True)
                         if args.resume else None)
    init_state = None
    if args.init_checkpoint:
        init_state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if init_state["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("initialization checkpoint tokenizer hash does not match")
        if init_state["config"] != config.__dict__:
            raise ValueError("initialization checkpoint model configuration does not match")
    if progressive_schedule and init_state is None:
        raise ValueError("progressive growth requires the full-depth initialization checkpoint")
    model = AncientDecoder(config)
    if init_state is not None:
        model.load_state_dict(init_state["model"], strict=True)
    if progressive_schedule:
        model.truncate_layers(int(progressive_schedule["stages"][0]["n_layers"]))
    model = model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = optimizer_factory(args.optimizer, model, args.lr, args.weight_decay)
    training_config = {"sequence_length": args.sequence_length,
                       "grad_accum": args.grad_accum, "lr": args.lr,
                       "weight_decay": args.weight_decay,
                       "sampling_mode": args.sampling_mode,
                       "curriculum": curriculum_data,
                       "depth_schedule": progressive_schedule,
                       "growth_stabilization_steps": args.growth_stabilization_steps}

    start_step = start_tokens = 0
    if resume_checkpoint:
        checkpoint = resume_checkpoint
        if checkpoint["optimizer_name"] != args.optimizer:
            raise ValueError("cannot resume with a different optimizer")
        if checkpoint["seed"] != args.seed:
            raise ValueError("resume checkpoint model seed differs")
        if checkpoint["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("resume checkpoint tokenizer hash differs")
        if checkpoint["sampling_weights"] != weights or checkpoint["data_seed"] != args.data_seed:
            raise ValueError("resume sampling configuration differs")
        if checkpoint.get("training_config") != training_config:
            raise ValueError("resume sequence/optimizer hyperparameters differ")
        saved_depth_state = checkpoint.get("depth_state")
        if bool(progressive_schedule) != bool(saved_depth_state and saved_depth_state.get("enabled")):
            raise ValueError("resume checkpoint progressive-depth mode differs")
        if progressive_schedule:
            if not saved_depth_state:
                raise ValueError("progressive resume checkpoint has no layer-growth state")
            for stage in progressive_schedule["stages"][1:saved_depth_state["stage_index"] + 1]:
                added = model.grow_layers(int(stage["n_layers"]), init_state["model"],
                                          device=device, dtype=torch.bfloat16)
                optimizer.add_param_group({"params": added})
        if checkpoint["config"] != model.config_dict():
            raise ValueError("resume checkpoint model configuration differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        start_step, start_tokens = int(checkpoint["step"]), int(checkpoint["tokens"])

    log_path = args.output / "train-log.jsonl"
    eval_path = args.output / "eval-log.jsonl"
    if not args.resume and (log_path.exists() or eval_path.exists()):
        raise FileExistsError("output already has training logs; choose a new directory or resume")
    if not args.resume:
        log_path.write_text("")
        eval_path.write_text("")

    record_pool = None
    sampler_state = None
    if curriculum_stages is None:
        records = weighted_records("train", weights, seed=args.data_seed)
    else:
        all_specs = sorted({spec for stage in curriculum_stages
                            for spec in stage["weights"]})
        record_pool = WeightedRecordPool("train", all_specs, seed=args.data_seed)
        sampler_state = {"weights": curriculum_stages[0]["weights"]}
        records = scheduled_records(record_pool, lambda: sampler_state["weights"])
    batches = iter(pack_records(records, tokenizer, sequence_length=args.sequence_length,
                                batch_size=1))
    digest = hashlib.sha256()
    steps_target = math.ceil(args.tokens / (args.sequence_length * args.grad_accum))
    replay_tokens = 0
    for _ in range(start_step):
        if record_pool is not None:
            replay_stage = curriculum_stage_index(curriculum_stages, replay_tokens, args.tokens)
            sampler_state["weights"] = curriculum_stages[replay_stage]["weights"]
        for _ in range(args.grad_accum):
            x, y = next(batches)
            update_data_digest(digest, x, y)
            replay_tokens += x.numel()
    if start_tokens and start_tokens != replay_tokens:
        raise ValueError("resume token count does not match deterministic stream replay")

    step_end = steps_target
    if args.stop_after_steps is not None:
        step_end = min(step_end, args.stop_after_steps)
    processed_tokens = start_tokens
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    losses_window = []
    validation_history = (list(resume_checkpoint.get("validation_history", []))
                          if resume_checkpoint else [])
    curriculum_stage_history = []
    last_stage_index = None
    latest_path = args.output / "latest-checkpoint.pt"

    if resume_checkpoint:
        depth_state = resume_checkpoint.get("depth_state")
    else:
        depth_state = None
    if depth_state is None:
        depth_state = {
            "enabled": bool(progressive_schedule),
            "stage_index": 0,
            "stage_history": [{"step": start_step + 1, "tokens": start_tokens,
                               "n_layers": model.config.n_layers}],
            "expansion_events": [], "completed_stages": [],
            "current_stage": fresh_stage_metrics(model, start_tokens, start_step + 1),
            "stabilization_remaining": 0,
            "overall_peak_vram_allocated_bytes": 0,
            "overall_peak_vram_reserved_bytes": 0,
        }
    set_stabilization_trainable(model, int(depth_state["stage_index"]),
                                progressive_schedule["stages"] if progressive_schedule else [],
                                int(depth_state.get("stabilization_remaining", 0)))
    elapsed_offset = float(resume_checkpoint.get("elapsed_seconds", 0.0)) if resume_checkpoint else 0.0

    def elapsed_total():
        return elapsed_offset + time.perf_counter() - start_time

    def update_peak_metrics():
        if device.type != "cuda":
            return
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        stage_metrics = depth_state["current_stage"]
        stage_metrics["peak_vram_allocated_bytes"] = max(
            stage_metrics["peak_vram_allocated_bytes"], allocated)
        stage_metrics["peak_vram_reserved_bytes"] = max(
            stage_metrics["peak_vram_reserved_bytes"], reserved)
        depth_state["overall_peak_vram_allocated_bytes"] = max(
            depth_state["overall_peak_vram_allocated_bytes"], allocated)
        depth_state["overall_peak_vram_reserved_bytes"] = max(
            depth_state["overall_peak_vram_reserved_bytes"], reserved)

    def save_current_checkpoint(path, step, tokens):
        update_peak_metrics()
        save_checkpoint(path, model, optimizer, model.config_dict(), args.seed,
                        args.optimizer, step, tokens, tokenizer_hash, weights,
                        args.data_seed,
                        str(args.init_checkpoint) if args.init_checkpoint else None,
                        training_config, depth_state=depth_state,
                        validation_history=validation_history,
                        elapsed_seconds=elapsed_total())

    pending_added = None
    try:
        for step in range(start_step + 1, step_end + 1):
            just_expanded = False
            if progressive_schedule:
                desired_stage = stage_for_tokens(progressive_schedule["stages"],
                                                 processed_tokens)
                while desired_stage > int(depth_state["stage_index"]):
                    previous_stage = int(depth_state["stage_index"])
                    next_stage = previous_stage + 1
                    before_loss = evaluate(model, tokenizer, device, languages,
                                           args.sequence_length, args.eval_batches)
                    before_time = elapsed_total()
                    update_peak_metrics()
                    closed = close_stage_metrics(
                        depth_state["current_stage"], processed_tokens, step - 1,
                        optimizer, device)
                    closed["estimated_training_flops"] = int(
                        depth_state["current_stage"]["estimated_training_flops"])
                    depth_state["completed_stages"].append(closed)

                    inherited_states = {param: state for param, state in optimizer.state.items()}
                    old_param_ids = {id(parameter) for parameter in model.parameters()}
                    old_group_count = len(optimizer.param_groups)
                    old_state_tensors = optimizer_state_tensor_count(optimizer)
                    added = model.grow_layers(int(progressive_schedule["stages"][next_stage]["n_layers"]),
                                              init_state["model"], device=device,
                                              dtype=torch.bfloat16)
                    optimizer.add_param_group({"params": added})
                    preserved = all(optimizer.state.get(param) is state
                                    for param, state in inherited_states.items())
                    if not preserved:
                        raise RuntimeError("optimizer state for inherited parameters changed during growth")
                    added_ids = {id(parameter) for parameter in added}
                    if added_ids & old_param_ids:
                        raise RuntimeError("new layer unexpectedly reused an inherited parameter")
                    if len(optimizer.param_groups) != old_group_count + 1:
                        raise RuntimeError("new layer optimizer group was not added")
                    event = {
                        "step": step, "tokens": processed_tokens,
                        "from_layers": int(progressive_schedule["stages"][previous_stage]["n_layers"]),
                        "to_layers": model.config.n_layers,
                        "parameters_before": closed["parameters"],
                        "parameters_after": model_parameter_count(model.config),
                        "validation_before": before_loss,
                        "validation_mean_before": mean_validation_loss(before_loss),
                        "validation_before_elapsed_seconds": before_time,
                        "optimizer_state_entries_preserved": len(inherited_states),
                        "optimizer_state_tensors_before": old_state_tensors,
                        "optimizer_groups_before": old_group_count,
                        "optimizer_groups_after": len(optimizer.param_groups),
                        "new_parameter_count": sum(parameter.numel() for parameter in added),
                        "new_parameter_state_initialized_before_first_step": sum(
                            parameter in optimizer.state for parameter in added),
                        "identity_residual_projections_zeroed": True,
                        "stabilization_steps": args.growth_stabilization_steps,
                    }
                    after_loss = evaluate(model, tokenizer, device, languages,
                                          args.sequence_length, args.eval_batches)
                    event["validation_after"] = after_loss
                    event["validation_mean_after"] = mean_validation_loss(after_loss)
                    event["validation_delta"] = (
                        event["validation_mean_after"] - event["validation_mean_before"]
                        if event["validation_mean_after"] is not None
                        and event["validation_mean_before"] is not None else None)
                    event["validation_after_elapsed_seconds"] = elapsed_total()
                    depth_state["expansion_events"].append(event)
                    depth_state["stage_index"] = next_stage
                    depth_state["stage_history"].append({
                        "step": step, "tokens": processed_tokens,
                        "n_layers": model.config.n_layers})
                    depth_state["stabilization_remaining"] = args.growth_stabilization_steps
                    depth_state["current_stage"] = fresh_stage_metrics(
                        model, processed_tokens, step)
                    set_stabilization_trainable(
                        model, next_stage, progressive_schedule["stages"],
                        int(depth_state["stabilization_remaining"]))
                    pending_added = {"parameters": added,
                                     "snapshots": [parameter.detach().clone() for parameter in added],
                                     "event": event, "steps": 0}
                    just_expanded = True
                    update_peak_metrics()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    desired_stage = stage_for_tokens(progressive_schedule["stages"],
                                                     processed_tokens)

            if record_pool is not None:
                stage_index = curriculum_stage_index(curriculum_stages,
                                                     processed_tokens, args.tokens)
                sampler_state["weights"] = curriculum_stages[stage_index]["weights"]
                if stage_index != last_stage_index:
                    stage_record = {"step": step, "tokens": processed_tokens,
                                    "stage": curriculum_stages[stage_index]["name"]}
                    curriculum_stage_history.append(stage_record)
                    last_stage_index = stage_index
            stabilization_step = int(depth_state.get("stabilization_remaining", 0)) > 0
            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            for _ in range(args.grad_accum):
                x, y = next(batches)
                update_data_digest(digest, x, y)
                out = model(x.to(device), y.to(device))
                loss = out["loss"]
                (loss / args.grad_accum).backward()
                losses_window.append(float(loss.detach()))
                processed_tokens += x.numel()
                step_tokens += x.numel()
            check_gradients = (step == start_step + 1 or step % 10 == 0
                               or step == step_end or just_expanded
                               or stabilization_step)
            if check_gradients:
                finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                             for p in model.parameters())
                if not finite:
                    raise RuntimeError(f"non-finite gradient at optimizer step {step}")
            optimizer.step()
            update_seconds = time.perf_counter() - update_start
            current_stage = depth_state["current_stage"]
            current_stage["tokens"] += step_tokens
            current_stage["optimizer_steps"] += 1
            current_stage["training_compute_seconds"] += update_seconds
            current_stage["estimated_training_flops"] += estimate_training_flops(
                model.config, step_tokens, args.sequence_length)
            if pending_added is not None:
                added_parameters = pending_added["parameters"]
                grads_finite = all(parameter.grad is not None
                                   and torch.isfinite(parameter.grad).all().item()
                                   for parameter in added_parameters)
                changed = sum(not torch.equal(parameter.detach(), initial)
                              for parameter, initial in zip(
                                  added_parameters, pending_added["snapshots"]))
                state_count = sum(parameter in optimizer.state
                                  and any(torch.is_tensor(value)
                                          for value in optimizer.state[parameter].values())
                                  for parameter in added_parameters)
                if not grads_finite or changed == 0:
                    raise RuntimeError("new layer failed finite-gradient/update verification")
                pending_added["steps"] += 1
                if pending_added["steps"] == 1:
                    pending_added["event"]["added_layer_first_step"] = {
                        "step": step, "finite_gradients": grads_finite,
                        "parameters_with_optimizer_state": int(state_count),
                        "parameter_tensors_changed": int(changed),
                        "parameter_tensors_total": len(added_parameters),
                    }
                if (not stabilization_step
                        or int(depth_state["stabilization_remaining"]) == 1):
                    changed_after_stabilization = sum(
                        not torch.equal(parameter.detach(), initial)
                        for parameter, initial in zip(
                            added_parameters, pending_added["snapshots"]))
                    pending_added["event"]["added_layer_after_stabilization"] = {
                        "step": step,
                        "optimizer_steps_since_insertion": pending_added["steps"],
                        "finite_gradients": grads_finite,
                        "parameters_with_optimizer_state": int(state_count),
                        "parameter_tensors_changed": int(changed_after_stabilization),
                        "parameter_tensors_total": len(added_parameters),
                    }
                    if changed_after_stabilization == 0:
                        raise RuntimeError("new layer did not change during stabilization")
                    pending_added = None
            if stabilization_step:
                depth_state["stabilization_remaining"] -= 1
                if depth_state["stabilization_remaining"] == 0:
                    for parameter in model.parameters():
                        parameter.requires_grad_(True)
            if step == start_step + 1 or step % 10 == 0 or step == step_end:
                rec = {"step": step, "tokens": processed_tokens,
                       "train_loss": sum(losses_window) / len(losses_window),
                       "elapsed_seconds": elapsed_total(),
                       "n_layers": model.config.n_layers,
                       "stabilizing_new_layers": stabilization_step}
                losses_window.clear()
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            should_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step == step_end)
            if should_eval and step > start_step:
                per_lang = evaluate(model, tokenizer, device, languages,
                                    args.sequence_length, args.eval_batches)
                validation_history.append({"step": step, "tokens": processed_tokens,
                                           "elapsed_seconds": elapsed_total(),
                                           "n_layers": model.config.n_layers,
                                           "by_language": per_lang})
                with eval_path.open("a") as f:
                    f.write(json.dumps(validation_history[-1]) + "\n")
            update_peak_metrics()
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_current_checkpoint(latest_path, step, processed_tokens)
    except StopIteration as exc:
        raise RuntimeError(f"training stream exhausted after {processed_tokens} tokens") from exc

    final_step = step_end
    is_partial = final_step < steps_target
    invocation_elapsed = time.perf_counter() - start_time
    elapsed = elapsed_offset + invocation_elapsed
    update_peak_metrics()
    partial_stage = close_stage_metrics(depth_state["current_stage"],
                                        processed_tokens, final_step, optimizer, device)
    partial_stage["estimated_training_flops"] = int(
        depth_state["current_stage"]["estimated_training_flops"])
    stage_metrics = list(depth_state["completed_stages"]) + [partial_stage]
    peak_alloc = int(depth_state["overall_peak_vram_allocated_bytes"])
    peak_reserved = int(depth_state["overall_peak_vram_reserved_bytes"])
    save_current_checkpoint(latest_path, final_step, processed_tokens)
    final_path = args.output / "final-checkpoint.pt"
    if not is_partial:
        save_current_checkpoint(final_path, final_step, processed_tokens)

    state_bytes = optimizer_state_bytes(optimizer)
    reloaded = AncientDecoder(ModelConfig(**model.config_dict())).to(
        device=device, dtype=torch.bfloat16)
    reloaded.load_state_dict(model.state_dict())
    generations = ({} if is_partial or args.skip_generation else
                   generate_examples(reloaded, tokenizer, device, languages))
    report = {
        "status": "checkpointed_partial" if is_partial else "completed",
        "config": model.config_dict(), "parameters": parameter_count(ModelConfig(**model.config_dict())),
        "target_parameters": parameter_count(config),
        "optimizer": args.optimizer, "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "sampling_mode": args.sampling_mode,
        "curriculum_name": curriculum_data.get("name") if curriculum_data else None,
        "curriculum_stage_history": curriculum_stage_history,
        "depth_schedule": progressive_schedule,
        "depth_stage_history": depth_state["stage_history"],
        "layer_expansion_events": depth_state["expansion_events"],
        "depth_stage_metrics": stage_metrics,
        "sampled_records_by_source_language": ({
            f"{source}:{lang}": count
            for (source, lang), count in sorted(record_pool.records_by_source_language.items())
        } if record_pool is not None else None),
        "sequence_length": args.sequence_length, "microbatch": 1,
        "grad_accum": args.grad_accum, "target_tokens": args.tokens,
        "actual_tokens": processed_tokens, "optimizer_steps": final_step,
        "train_log": str(log_path), "eval_log": str(eval_path),
        "final_checkpoint": str(final_path) if not is_partial else None,
        "latest_checkpoint": str(latest_path),
        "validation_history": validation_history,
        "validation_loss_by_language": validation_history[-1]["by_language"] if validation_history else None,
        "generations": generations, "tokenizer_sha256": tokenizer_hash,
        "initialization_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "initialization_checkpoint_sha256": sha256(args.init_checkpoint) if args.init_checkpoint else None,
        "sampling_weights": weights, "data_seed": args.data_seed,
        "training_batch_sha256": digest.hexdigest(), "seed": args.seed,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
        "elapsed_seconds_this_invocation": invocation_elapsed,
        "elapsed_seconds_total": elapsed,
        "tokens_per_second_this_invocation": (processed_tokens - start_tokens) / max(invocation_elapsed, 1e-9),
        "tokens_per_second_overall": processed_tokens / max(elapsed, 1e-9),
        "training_compute_seconds_total": sum(
            item["training_compute_seconds"] for item in stage_metrics),
        "estimated_training_flops": sum(
            item["estimated_training_flops"] for item in stage_metrics),
        "peak_vram_allocated_bytes": peak_alloc,
        "peak_vram_reserved_bytes": peak_reserved,
        "optimizer_state_memory": state_bytes,
        "checkpoint_reloaded_for_generation": not is_partial,
        "generation_skipped": args.skip_generation,
        "resume_source": str(args.resume) if args.resume else None,
    }
    report_path = args.output / ("partial-report.json" if is_partial else "training-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "optimizer", "parameters",
          "actual_tokens", "optimizer_steps", "validation_loss_by_language",
          "tokens_per_second_this_invocation", "peak_vram_allocated_bytes",
          "optimizer_state_memory", "training_batch_sha256",
          "depth_stage_history", "estimated_training_flops")}, indent=2))


if __name__ == "__main__":
    main()
