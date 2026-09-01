# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import regex as re
import torch

EXPERT_ADAPTER_SCHEMA = "qwen3_moe_expert_adapter_v1"

_MLP_MODULE_PATTERN = r"((?:language_model\.)?model\.layers\.\d+\.mlp)"
_GATE_UP = re.compile(
    rf"^{_MLP_MODULE_PATTERN}\.shared_expert\.trainable_expert"
    r"\.gate_up_proj\.weight$"
)
_DOWN = re.compile(
    rf"^{_MLP_MODULE_PATTERN}\.shared_expert\.trainable_expert"
    r"\.down_proj\.weight$"
)
_ROUTER = re.compile(rf"^{_MLP_MODULE_PATTERN}\.gate\.trainable_expert_router\.weight$")
_BIAS = re.compile(rf"^{_MLP_MODULE_PATTERN}\.shared_expert\.expert_bias$")
_PATTERNS = (_GATE_UP, _DOWN, _ROUTER, _BIAS)


@dataclass(frozen=True)
class StructuralExpertWeights:
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    router: torch.Tensor | None
    expert_bias: torch.Tensor

    @property
    def is_routed(self) -> bool:
        return self.router is not None


def is_structural_expert_tensor_name(name: str) -> bool:
    return any(pattern.fullmatch(name) is not None for pattern in _PATTERNS)


def route_structural_expert(
    gating_output: torch.Tensor,
    extra_logits: torch.Tensor,
    active: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route over base experts plus one request-scoped expert candidate."""
    num_base_experts = gating_output.shape[-1]
    extra_logits = extra_logits.masked_fill(~active, -torch.inf)
    combined_logits = torch.cat((gating_output, extra_logits.unsqueeze(-1)), dim=-1)
    if scoring_func == "softmax":
        scores = torch.softmax(combined_logits.float(), dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(combined_logits.float())
    else:
        raise ValueError(f"Unsupported structural expert scoring: {scoring_func}")

    choice_scores = scores.clone()
    choice_scores[:, :num_base_experts] += base_bias
    topk_ids = choice_scores.topk(topk, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_weights *= routed_scaling_factor

    added = topk_ids == num_base_experts
    added_weights = torch.where(
        added, topk_weights, torch.zeros_like(topk_weights)
    ).sum(dim=-1)
    base_ids = torch.where(added, torch.full_like(topk_ids, -1), topk_ids)
    base_weights = torch.where(added, torch.zeros_like(topk_weights), topk_weights)
    return base_weights, base_ids, added_weights


def extract_structural_experts(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, StructuralExpertWeights]]:
    """Split portable structural-expert tensors from ordinary LoRA tensors."""
    ordinary: dict[str, torch.Tensor] = {}
    parts: dict[str, dict[str, torch.Tensor]] = {}
    patterns = (
        (_GATE_UP, "gate_up_proj"),
        (_DOWN, "down_proj"),
        (_ROUTER, "router"),
        (_BIAS, "expert_bias"),
    )

    for name, tensor in tensors.items():
        for pattern, part_name in patterns:
            match = pattern.fullmatch(name)
            if match is not None:
                parts.setdefault(match.group(1), {})[part_name] = tensor
                break
        else:
            ordinary[name] = tensor

    experts: dict[str, StructuralExpertWeights] = {}
    for module_name, module_parts in parts.items():
        missing = {"gate_up_proj", "down_proj", "expert_bias"} - module_parts.keys()
        if missing:
            raise ValueError(
                f"Structural expert {module_name} is missing tensors: {sorted(missing)}"
            )
        gate_up = module_parts["gate_up_proj"]
        down = module_parts["down_proj"]
        if gate_up.ndim != 2 or gate_up.shape[0] % 2:
            raise ValueError(
                f"Structural expert {module_name} gate_up_proj must have shape "
                f"(2 * intermediate_size, hidden_size), got {tuple(gate_up.shape)}"
            )
        intermediate_size = gate_up.shape[0] // 2
        if down.shape != (gate_up.shape[1], intermediate_size):
            raise ValueError(
                f"Structural expert {module_name} down_proj has shape "
                f"{tuple(down.shape)}, expected {(gate_up.shape[1], intermediate_size)}"
            )
        router = module_parts.get("router")
        if router is not None and router.shape != (1, gate_up.shape[1]):
            raise ValueError(
                f"Structural expert {module_name} router has shape "
                f"{tuple(router.shape)}, expected {(1, gate_up.shape[1])}"
            )
        expert_bias = module_parts["expert_bias"]
        if expert_bias.ndim != 1:
            raise ValueError(
                f"Structural expert {module_name} expert_bias must be one-dimensional"
            )
        experts[module_name] = StructuralExpertWeights(
            gate_up_proj=gate_up,
            down_proj=down,
            router=router,
            expert_bias=expert_bias,
        )

    return ordinary, experts
