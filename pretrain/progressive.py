"""Small helpers for fixed-stage progressive transformer depth growth."""
from __future__ import annotations

import json
from pathlib import Path

from .model import parameter_count


def load_depth_schedule(path: Path, *, final_depth: int, target_tokens: int):
    schedule = json.loads(path.read_text())
    stages = schedule.get("stages", [])
    if not stages or stages[0].get("start_tokens") != 0:
        raise ValueError("depth schedule must start at token 0")
    previous_tokens = -1
    previous_depth = 0
    for stage in stages:
        tokens = int(stage["start_tokens"])
        depth = int(stage["n_layers"])
        if tokens <= previous_tokens or tokens < 0:
            raise ValueError("depth stage token milestones must strictly increase")
        if depth <= previous_depth:
            raise ValueError("depth stages must strictly increase layer count")
        if tokens > target_tokens:
            raise ValueError("depth stage starts after the requested training budget")
        previous_tokens, previous_depth = tokens, depth
    if previous_depth != final_depth:
        raise ValueError(f"last depth {previous_depth} does not match target {final_depth}")
    return schedule


def stage_for_tokens(stages, tokens: int) -> int:
    index = 0
    for i, stage in enumerate(stages):
        if int(stage["start_tokens"]) <= tokens:
            index = i
        else:
            break
    return index


def model_parameter_count(config) -> int:
    """Parameter count for the current-depth model configuration."""
    return parameter_count(config)


def estimate_training_flops(config, tokens: int, sequence_length: int | None = None) -> int:
    """Approximate training FLOPs: 6*parameters/token plus causal attention."""
    params = model_parameter_count(config)
    sequence_length = sequence_length or config.max_seq_len
    attention_per_token = 6 * config.n_layers * config.d_model * sequence_length
    return int(tokens * (6 * params + attention_per_token))
