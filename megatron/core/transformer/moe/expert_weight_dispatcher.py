# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert-weight dispatch for BalancedMoELayer.

The dispatcher moves checkpoint-owned home expert weights into runtime spare
slots. Forward gathers home weights across the EP group and selects the weights
needed by the local spare slots. Backward explicitly folds spare gradients back
to the owning home expert parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class ExpertWeightDispatchMetadata:
    """Routing metadata produced from a home-expert to spare-slot map."""

    global_routing_map: torch.Tensor
    local_to_global_routing_map: torch.Tensor
    global_to_local_routing_map: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]
    num_out_experts: int
    has_experts_per_slot: torch.Tensor
    local_spare_home_indices: torch.Tensor


class _AllToAllExpertWeightDispatch(torch.autograd.Function):
    """Autograd bridge for home-weight to spare-slot dispatch."""

    @staticmethod
    def forward(
        ctx,
        expert_offloading_map: torch.Tensor,
        ep_group: torch.distributed.ProcessGroup,
        ep_rank: int,
        ep_size: int,
        num_local_home_experts: int,
        num_local_spare_experts: int,
        *local_home_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(local_home_weights) != num_local_home_experts:
            raise ValueError(
                f"Expected {num_local_home_experts} local home weights, "
                f"got {len(local_home_weights)}."
            )
        if not local_home_weights:
            raise ValueError("At least one local home weight is required.")

        reference_weight = local_home_weights[0]
        for weight in local_home_weights:
            if weight.shape != reference_weight.shape:
                raise ValueError(
                    "All expert weights in one dispatch call must have the same shape; "
                    f"got {tuple(weight.shape)} and {tuple(reference_weight.shape)}."
                )

        local_stack = torch.stack(local_home_weights, dim=0)
        gathered_stacks = [torch.empty_like(local_stack) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered_stacks, local_stack, group=ep_group)
        global_home_weights = torch.cat(gathered_stacks, dim=0)

        local_spare_home_indices: list[int] = []
        local_spare_start = ep_rank * num_local_spare_experts
        outputs: list[torch.Tensor] = []
        for local_spare_idx in range(num_local_spare_experts):
            spare_idx = local_spare_start + local_spare_idx
            home_indices = torch.where(expert_offloading_map[:, spare_idx])[0]
            if home_indices.numel() == 0:
                local_spare_home_indices.append(-1)
                outputs.append(torch.zeros_like(reference_weight))
            elif home_indices.numel() == 1:
                home_idx = int(home_indices.item())
                local_spare_home_indices.append(home_idx)
                outputs.append(global_home_weights[home_idx])
            else:
                raise ValueError(f"Spare slot {spare_idx} maps to more than one home expert.")

        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size
        ctx.num_home_experts = expert_offloading_map.shape[0]
        ctx.num_local_home_experts = num_local_home_experts
        ctx.weight_shape = tuple(reference_weight.shape)
        ctx.weight_dtype = reference_weight.dtype
        ctx.weight_device = reference_weight.device
        ctx.local_spare_home_indices = local_spare_home_indices

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        global_home_grads = torch.zeros(
            (ctx.num_home_experts, *ctx.weight_shape),
            dtype=ctx.weight_dtype,
            device=ctx.weight_device,
        )

        for home_idx, grad_output in zip(ctx.local_spare_home_indices, grad_outputs):
            if home_idx < 0 or grad_output is None:
                continue
            global_home_grads[home_idx].add_(grad_output)

        torch.distributed.all_reduce(global_home_grads, group=ctx.ep_group)
        local_home_start = ctx.ep_rank * ctx.num_local_home_experts
        local_home_end = local_home_start + ctx.num_local_home_experts
        local_home_grads = tuple(global_home_grads[local_home_start:local_home_end].unbind(0))

        return (None, None, None, None, None, None, *local_home_grads)


class AllToAllExpertWeightDispatcher:
    """Dispatch home expert weights to local spare expert slots across an EP group."""

    def __init__(
        self,
        config: TransformerConfig,
        ep_group: torch.distributed.ProcessGroup,
        num_home_experts: int,
        num_spare_experts: int,
    ) -> None:
        self.config = config
        self.ep_group = ep_group
        self.ep_size = torch.distributed.get_world_size(group=ep_group)
        self.ep_rank = torch.distributed.get_rank(group=ep_group)
        self.num_home_experts = num_home_experts
        self.num_spare_experts = num_spare_experts

        if self.num_home_experts <= 0:
            raise ValueError("num_home_experts must be positive.")
        if self.num_spare_experts <= 0:
            raise ValueError("num_spare_experts must be positive.")
        if self.num_home_experts % self.ep_size != 0:
            raise ValueError("num_home_experts must be divisible by EP size.")
        if self.num_spare_experts % self.ep_size != 0:
            raise ValueError("num_spare_experts must be divisible by EP size.")

        self.num_local_home_experts = self.num_home_experts // self.ep_size
        self.num_local_spare_experts = self.num_spare_experts // self.ep_size

    def preprocess(self, expert_offloading_map: torch.Tensor) -> ExpertWeightDispatchMetadata:
        """Validate and reshape the global home-to-spare expert map."""

        if expert_offloading_map.dtype != torch.bool:
            raise ValueError("expert_offloading_map must be a bool tensor.")
        expected_shape = (self.num_home_experts, self.num_spare_experts)
        if tuple(expert_offloading_map.shape) != expected_shape:
            raise ValueError(
                f"Expected expert_offloading_map shape {expected_shape}, "
                f"got {tuple(expert_offloading_map.shape)}."
            )
        if expert_offloading_map.sum(dim=0).max().item() > 1:
            raise ValueError("Each spare expert slot may map to at most one home expert.")

        home_start = self.ep_rank * self.num_local_home_experts
        home_end = home_start + self.num_local_home_experts
        spare_start = self.ep_rank * self.num_local_spare_experts
        spare_end = spare_start + self.num_local_spare_experts

        local_to_global = expert_offloading_map[home_start:home_end, :].reshape(
            self.num_local_home_experts, self.ep_size, self.num_local_spare_experts
        )
        global_to_local = expert_offloading_map[:, spare_start:spare_end].reshape(
            self.ep_size, self.num_local_home_experts, self.num_local_spare_experts
        )

        local_spare_home_indices = torch.full(
            (self.num_local_spare_experts,),
            -1,
            dtype=torch.long,
            device=expert_offloading_map.device,
        )
        for local_spare_idx in range(self.num_local_spare_experts):
            spare_idx = spare_start + local_spare_idx
            home_indices = torch.where(expert_offloading_map[:, spare_idx])[0]
            if home_indices.numel() == 1:
                local_spare_home_indices[local_spare_idx] = home_indices[0]

        return ExpertWeightDispatchMetadata(
            global_routing_map=expert_offloading_map,
            local_to_global_routing_map=local_to_global,
            global_to_local_routing_map=global_to_local,
            input_splits=local_to_global.sum(dim=(0, 2)).tolist(),
            output_splits=global_to_local.sum(dim=(1, 2)).tolist(),
            num_out_experts=int(local_to_global.sum().item()),
            has_experts_per_slot=global_to_local.sum(dim=(0, 1)),
            local_spare_home_indices=local_spare_home_indices,
        )

    def dispatch(
        self, metadata: ExpertWeightDispatchMetadata, *expert_weights: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return weights for this rank's local spare slots."""

        outputs = _AllToAllExpertWeightDispatch.apply(
            metadata.global_routing_map,
            self.ep_group,
            self.ep_rank,
            self.ep_size,
            self.num_local_home_experts,
            self.num_local_spare_experts,
            *expert_weights,
        )
        return list(outputs)

    def expert_dispatch(
        self, metadata: ExpertWeightDispatchMetadata, *expert_weights: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compatibility alias for the old Echo dispatcher method name."""

        return self.dispatch(metadata, *expert_weights)
