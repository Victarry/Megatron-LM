# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime expert offloading planner for BalancedMoELayer.

This module keeps the public Echo planner API while trimming the old exploratory
CLI-only code. The planner operates on home-expert routing maps, assigns
spillover tokens to spare expert slots, and returns an effective home-plus-spare
routing view expected by ``MoEAlltoAllTokenDispatcher``.
"""

import os
from typing import Literal, Union

import torch


AssignmentAlgorithm = Literal["one_shot_greedy", "approx_bin_packing"]


def _validate_common_inputs(
    map_token_to_expert: torch.Tensor,
    probs_routing: torch.Tensor,
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int,
) -> None:
    if map_token_to_expert.dtype is not torch.bool:
        raise ValueError("offloading_planner expects map_token_to_expert to be a bool tensor.")
    if map_token_to_expert.shape != probs_routing.shape:
        raise ValueError(
            "offloading_planner expects map_token_to_expert and probs_routing "
            f"to have the same shape, got {map_token_to_expert.shape} and {probs_routing.shape}."
        )
    if count_tokens_per_expert_from_ep_rank.ndim != 2:
        raise ValueError("offloading_planner expects token counts shaped [ep, num_experts].")
    if count_tokens_per_expert_from_ep_rank.shape[0] != num_ep_ranks:
        raise ValueError(
            "offloading_planner count tensor EP dimension does not match num_ep_ranks: "
            f"{count_tokens_per_expert_from_ep_rank.shape[0]} vs {num_ep_ranks}."
        )
    if count_tokens_per_expert_from_ep_rank.shape[1] != map_token_to_expert.shape[1]:
        raise ValueError(
            "offloading_planner count tensor expert dimension does not match routing map: "
            f"{count_tokens_per_expert_from_ep_rank.shape[1]} vs {map_token_to_expert.shape[1]}."
        )
    if map_token_to_expert.shape[1] % num_ep_ranks != 0:
        raise ValueError("offloading_planner requires num_experts divisible by num_ep_ranks.")
    if num_spare_experts_per_ep_rank <= 0:
        raise ValueError("offloading_planner requires a positive spare expert count per EP rank.")


def _ep_rank_to_int(ep_rank: Union[torch.Tensor, int]) -> int:
    if isinstance(ep_rank, torch.Tensor):
        return int(ep_rank.item())
    return int(ep_rank)


def _to_effective_order(
    home_first_tensor: torch.Tensor,
    num_home_experts: int,
    num_spare_experts: int,
    num_ep_ranks: int,
) -> torch.Tensor:
    home = home_first_tensor[:, :num_home_experts].reshape(
        -1, num_ep_ranks, num_home_experts // num_ep_ranks
    )
    spare = home_first_tensor[:, num_home_experts:].reshape(
        -1, num_ep_ranks, num_spare_experts // num_ep_ranks
    )
    return torch.cat([home, spare], dim=-1).reshape(
        -1, num_home_experts + num_spare_experts
    )


def one_shot_greedy_assignment(
    count_tokens_per_chunk: torch.Tensor,
    capacity_per_bucket: torch.Tensor,
) -> torch.Tensor:
    """Assign token chunks to capacity buckets by cumulative interval overlap."""

    count_tokens_per_chunk_cumsum = torch.cumsum(count_tokens_per_chunk, dim=0)
    capacity_per_bucket_cumsum = torch.cumsum(capacity_per_bucket, dim=0)
    count_tokens_per_chunk_start = count_tokens_per_chunk_cumsum - count_tokens_per_chunk
    capacity_per_bucket_start = capacity_per_bucket_cumsum - capacity_per_bucket

    chunk_start = count_tokens_per_chunk_start.unsqueeze(1)
    chunk_end = count_tokens_per_chunk_cumsum.unsqueeze(1)
    bucket_start = capacity_per_bucket_start.unsqueeze(0)
    bucket_end = capacity_per_bucket_cumsum.unsqueeze(0)

    overlap_start = torch.maximum(chunk_start, bucket_start)
    overlap_end = torch.minimum(chunk_end, bucket_end)
    return (overlap_end - overlap_start).clamp(min=0)


def approx_bin_packing_triton(
    count_spillover_per_expert: torch.Tensor,
    capacity_spare_per_ep_rank: torch.Tensor,
    avg_tokens_per_ep_rank: torch.Tensor,
    num_buckets: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate bin packing assignment.

    The old Echo implementation used a Triton kernel with bucketized search. For
    the MVP port we keep the public function API and greedy sorted assignment
    semantics, while relying on eager tensor updates for debuggability. The
    planner still routes spillover from larger experts into spare EP capacity in
    sorted order and returns the same assignment matrix contract.
    """

    del avg_tokens_per_ep_rank, num_buckets
    device = count_spillover_per_expert.device
    dtype = torch.int32
    num_experts = count_spillover_per_expert.shape[0]
    num_ep_ranks = capacity_spare_per_ep_rank.shape[0]

    remaining = count_spillover_per_expert.to(dtype).clone()
    capacity_remaining = capacity_spare_per_ep_rank.to(dtype).clone()
    assignment = torch.zeros(num_experts, num_ep_ranks, dtype=dtype, device=device)

    for expert_idx in range(num_experts):
        spillover = int(remaining[expert_idx].item())
        if spillover <= 0:
            continue
        for ep_idx in range(num_ep_ranks):
            capacity = int(capacity_remaining[ep_idx].item())
            if capacity <= 0:
                continue
            placed = min(spillover, capacity)
            if placed <= 0:
                continue
            assignment[expert_idx, ep_idx] = placed
            remaining[expert_idx] -= placed
            capacity_remaining[ep_idx] -= placed
            spillover -= placed
            if spillover <= 0:
                break

    return assignment, remaining


def reclaim_spare_experts(
    count_tokens_per_ep_rank: torch.Tensor,
    avg_tokens_per_ep_rank: torch.Tensor,
    count_tokens_from_home_expert_to_spare_expert: torch.Tensor,
    threshold_multiplier: float,
) -> torch.Tensor:
    """Disable spare assignments that are unnecessary under a load threshold."""

    num_ep_ranks = count_tokens_per_ep_rank.shape[0]
    num_home_experts, _ = count_tokens_from_home_expert_to_spare_expert.shape
    count_tokens_per_home_rank = count_tokens_per_ep_rank.view(num_ep_ranks, -1).sum(dim=1)
    threshold = threshold_multiplier * avg_tokens_per_ep_rank
    max_allowed_load = avg_tokens_per_ep_rank + threshold

    count_tokens_after_offloading = count_tokens_per_home_rank - (
        count_tokens_from_home_expert_to_spare_expert.sum(dim=1)
        .view(num_ep_ranks, -1)
        .sum(dim=1)
    )

    ep_rank_view = count_tokens_from_home_expert_to_spare_expert.view(
        num_ep_ranks, num_home_experts // num_ep_ranks, -1
    ).sum(dim=1)
    count_tokens_sorted, indices_sorted = torch.sort(ep_rank_view, dim=1)
    count_tokens_cumsum = torch.cumsum(count_tokens_sorted, dim=1)

    capacity_remaining = max_allowed_load - count_tokens_after_offloading
    idx_safe_steps = torch.searchsorted(
        count_tokens_cumsum, capacity_remaining.unsqueeze(1), side="right"
    ).squeeze(1)

    indices_steps = torch.arange(
        count_tokens_sorted.shape[1], device=count_tokens_from_home_expert_to_spare_expert.device
    ).unsqueeze(0)
    indices_steps = indices_steps.expand(count_tokens_sorted.shape[0], -1)
    mask_sorted = indices_steps >= idx_safe_steps.unsqueeze(1)

    mask_original_order = torch.zeros_like(mask_sorted)
    mask_original_order.scatter_(1, indices_sorted, mask_sorted)
    mask_column = mask_original_order.any(dim=0)

    return torch.where(
        mask_column.unsqueeze(0),
        count_tokens_from_home_expert_to_spare_expert,
        torch.zeros_like(count_tokens_from_home_expert_to_spare_expert),
    )


def _select_unique_home_per_spare(
    count_tokens_from_home_expert_to_spare_expert: torch.Tensor,
) -> torch.Tensor:
    """Keep the largest home-expert assignment for each runtime spare slot."""

    selected_counts, selected_home_indices = count_tokens_from_home_expert_to_spare_expert.max(
        dim=0
    )
    selected_spare_indices = torch.arange(
        count_tokens_from_home_expert_to_spare_expert.shape[1],
        device=count_tokens_from_home_expert_to_spare_expert.device,
    )
    active_spares = selected_counts > 0

    unique_assignment = torch.zeros_like(count_tokens_from_home_expert_to_spare_expert)
    unique_assignment[
        selected_home_indices[active_spares], selected_spare_indices[active_spares]
    ] = selected_counts[active_spares]
    return unique_assignment


def gen_intermediate(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate spillover tokens and spare capacity for planner assignment."""

    del ep_rank, threshold_multiplier
    count_tokens_per_expert = count_tokens_per_expert_from_ep_rank.sum(dim=0).to(dtype_index)
    count_tokens_per_ep_rank = count_tokens_per_expert.view(num_ep_ranks, -1).sum(dim=1)
    avg_tokens_per_ep_rank = count_tokens_per_ep_rank.sum() // num_ep_ranks
    deviation = count_tokens_per_ep_rank - avg_tokens_per_ep_rank
    capacity_spare_per_ep_rank = torch.relu(-deviation) * num_spare_experts_per_ep_rank

    local_counts = count_tokens_per_expert.view(num_ep_ranks, -1)
    count_tokens_per_local_expert_sorted, indices_local_expert_sorted = local_counts.sort(dim=1)
    spillover_cumsum = (
        count_tokens_per_local_expert_sorted.cumsum(dim=1) - avg_tokens_per_ep_rank
    ).clamp(min=0)
    count_spillover_per_expert_sorted = torch.cat(
        [spillover_cumsum[:, :1], torch.diff(spillover_cumsum, dim=1)], dim=1
    )
    count_spillover_per_home_expert = torch.scatter(
        torch.empty_like(count_spillover_per_expert_sorted),
        1,
        indices_local_expert_sorted,
        count_spillover_per_expert_sorted,
    ).view(-1)

    return count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank


def gen_assignment(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate home-expert to spare-expert assignment with one-shot greedy."""

    count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank = (
        gen_intermediate(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            num_spare_experts_per_ep_rank,
            threshold_multiplier,
            dtype_index,
        )
    )

    device = count_tokens_per_expert_from_ep_rank.device
    count_tokens_per_expert = count_tokens_per_expert_from_ep_rank.sum(dim=0).to(dtype_index)
    count_tokens_per_ep_rank = count_tokens_per_expert.view(num_ep_ranks, -1).sum(dim=1)

    count_spillover_sorted, indices_spillover_sort = torch.sort(
        count_spillover_per_home_expert, descending=True
    )
    capacity_spare_sorted, indices_spare_sort = torch.sort(
        capacity_spare_per_ep_rank, descending=True
    )
    count_tokens_from_chunk_to_bucket_sorted = one_shot_greedy_assignment(
        count_spillover_sorted, capacity_spare_sorted
    )
    count_spare_bucket_max, idx_spare_bucket_max = torch.topk(
        count_tokens_from_chunk_to_bucket_sorted, k=num_spare_experts_per_ep_rank, dim=0
    )
    num_buckets = count_tokens_from_chunk_to_bucket_sorted.shape[1]

    indices_row_sorted = idx_spare_bucket_max.transpose(0, 1).flatten()
    indices_row = indices_spillover_sort[indices_row_sorted]

    indices_ep_rank = torch.arange(num_buckets, device=device).repeat_interleave(
        num_spare_experts_per_ep_rank
    )
    indices_spare_slot = torch.arange(num_spare_experts_per_ep_rank, device=device).repeat(
        num_buckets
    )
    indices_ep_rank_original = indices_spare_sort[indices_ep_rank]
    indices_col = indices_ep_rank_original * num_spare_experts_per_ep_rank + indices_spare_slot
    count_tokens_values = count_spare_bucket_max.transpose(0, 1).flatten()

    count_tokens_from_home_expert_to_spare_expert = torch.zeros(
        count_spillover_per_home_expert.shape[0],
        num_buckets * num_spare_experts_per_ep_rank,
        device=device,
        dtype=count_tokens_from_chunk_to_bucket_sorted.dtype,
    )
    count_tokens_from_home_expert_to_spare_expert[indices_row, indices_col] = (
        count_tokens_values
    )

    if threshold_multiplier > 0:
        count_tokens_from_home_expert_to_spare_expert = reclaim_spare_experts(
            count_tokens_per_ep_rank,
            avg_tokens_per_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
            threshold_multiplier,
        )

    return (
        count_tokens_from_home_expert_to_spare_expert.to(dtype_index),
        count_spillover_per_home_expert,
        capacity_spare_per_ep_rank,
    )


def gen_assignment_for_approx_bp(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    dtype_index: torch.dtype = torch.int32,
    num_buckets: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate home-expert to spare-rank assignment for approx bin packing."""

    count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank = (
        gen_intermediate(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            num_spare_experts_per_ep_rank=1,
            threshold_multiplier=0.0,
            dtype_index=dtype_index,
        )
    )

    count_spillover_sorted, indices_spillover_sort = torch.sort(
        count_spillover_per_home_expert, descending=True
    )
    capacity_spare_sorted, indices_spare_sort = torch.sort(
        capacity_spare_per_ep_rank, descending=True
    )

    count_tokens_from_chunk_to_bucket_sorted, _ = approx_bin_packing_triton(
        count_spillover_sorted,
        capacity_spare_sorted,
        avg_tokens_per_ep_rank,
        num_buckets=num_buckets,
    )

    inverse_spillover_perm = torch.argsort(indices_spillover_sort)
    inverse_spare_perm = torch.argsort(indices_spare_sort)
    count_tokens_from_chunk_to_bucket = count_tokens_from_chunk_to_bucket_sorted[
        inverse_spillover_perm
    ][:, inverse_spare_perm]

    return (
        count_tokens_from_chunk_to_bucket.to(dtype_index),
        count_spillover_per_home_expert,
        capacity_spare_per_ep_rank,
    )


def breadth_first_allocation(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    count_tokens_from_home_expert_to_spare_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate tokens from each selected home expert proportionally by source EP rank."""

    device = count_tokens_per_expert_from_ep_rank.device
    counts = count_tokens_per_expert_from_ep_rank.to(device).float()
    assignment = count_tokens_from_home_expert_to_spare_expert.to(device).float()

    idx_supplier = assignment.argmax(0)
    mask_active = (assignment > 0).sum(0) > 0
    capacity = assignment[idx_supplier, torch.arange(assignment.shape[1], device=device)]

    count_tokens_rel = counts[:, idx_supplier]
    denominator = count_tokens_rel.sum(0, keepdim=True)
    probs_proportional = torch.where(
        denominator > 0, count_tokens_rel / denominator, torch.zeros_like(count_tokens_rel)
    )
    count_tokens_ideal = probs_proportional * capacity
    count_tokens_floors = torch.floor(count_tokens_ideal).int() * mask_active

    count_tokens_offloaded = torch.zeros_like(counts, dtype=torch.int32)
    idx_supplier_expanded = idx_supplier.unsqueeze(0).expand(counts.shape[0], -1)
    count_tokens_offloaded.scatter_add_(1, idx_supplier_expanded, count_tokens_floors)

    count_tokens_per_expert_after_offload = counts - count_tokens_offloaded
    capacity_spare_remaining = assignment.clone()
    capacity_spare_remaining[
        idx_supplier, torch.arange(assignment.shape[1], device=device)
    ] -= count_tokens_floors.sum(dim=0)

    return (
        count_tokens_floors,
        count_tokens_per_expert_after_offload,
        capacity_spare_remaining,
    )


def depth_first_allocation(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    capacity_spare_remaining: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate remaining spare capacity by depth-first cumulative overlap."""

    num_ep_ranks, num_experts = count_tokens_per_expert_from_ep_rank.shape
    num_experts_2, _ = capacity_spare_remaining.shape
    if num_experts != num_experts_2:
        raise ValueError("offloading_planner depth_first_allocation dimension mismatch.")

    count_tokens_cumsum = torch.cumsum(count_tokens_per_expert_from_ep_rank, dim=0)
    count_tokens_start = count_tokens_cumsum - count_tokens_per_expert_from_ep_rank
    count_tokens_end = count_tokens_cumsum

    capacity_cumsum = torch.cumsum(capacity_spare_remaining, dim=1)
    capacity_start = capacity_cumsum - capacity_spare_remaining
    capacity_end = capacity_cumsum

    overlap_start = torch.maximum(count_tokens_start.unsqueeze(2), capacity_start.unsqueeze(0))
    overlap_end = torch.minimum(count_tokens_end.unsqueeze(2), capacity_end.unsqueeze(0))
    count_tokens_overlap = (overlap_end - overlap_start).clamp(min=0)
    count_tokens_offloaded_from_expert_to_spare = count_tokens_overlap.sum(dim=1)

    idx_supplier = capacity_spare_remaining.argmax(0)
    idx_supplier_expanded = idx_supplier.unsqueeze(0).expand(num_ep_ranks, -1)
    count_tokens_after_second_offload = count_tokens_per_expert_from_ep_rank.scatter_add(
        1, idx_supplier_expanded, -count_tokens_offloaded_from_expert_to_spare
    )
    return count_tokens_offloaded_from_expert_to_spare, count_tokens_after_second_offload


def reroute_tokens_eager(
    map_token_to_expert: torch.Tensor,
    probs_routing: torch.Tensor,
    count_tokens_offloading_from_expert: torch.Tensor,
    count_tokens_offloading_to_spare: torch.Tensor,
    map_home_expert_to_spare: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move selected token assignments from home columns to spare columns."""

    del count_tokens_offloading_from_expert
    device = map_token_to_expert.device
    num_tokens, num_home_experts = map_token_to_expert.shape
    num_spare_experts = map_home_expert_to_spare.shape[1]

    rerouting_home_first = torch.zeros(
        num_tokens,
        num_home_experts + num_spare_experts,
        dtype=torch.bool,
        device=device,
    )
    rerouted_probs_home_first = torch.zeros(
        num_tokens,
        num_home_experts + num_spare_experts,
        dtype=probs_routing.dtype,
        device=device,
    )
    rerouting_home_first[:, :num_home_experts] = map_token_to_expert
    rerouted_probs_home_first[:, :num_home_experts] = probs_routing

    offset_by_home = [0 for _ in range(num_home_experts)]
    count_tokens_offloading_to_spare = count_tokens_offloading_to_spare.reshape(-1).to(torch.int64)

    for spare_idx in range(num_spare_experts):
        home_indices = torch.where(map_home_expert_to_spare[:, spare_idx])[0]
        if home_indices.numel() == 0:
            continue
        home_idx = int(home_indices[0].item())
        num_to_move = int(count_tokens_offloading_to_spare[spare_idx].item())
        if num_to_move <= 0:
            continue

        token_indices = torch.where(map_token_to_expert[:, home_idx])[0]
        start = offset_by_home[home_idx]
        end = min(start + num_to_move, token_indices.numel())
        if end <= start:
            continue

        tokens_to_move = token_indices[start:end]
        offset_by_home[home_idx] = end
        spare_col = num_home_experts + spare_idx

        rerouting_home_first[tokens_to_move, home_idx] = False
        rerouting_home_first[tokens_to_move, spare_col] = True
        rerouted_probs_home_first[tokens_to_move, spare_col] = rerouted_probs_home_first[
            tokens_to_move, home_idx
        ]
        rerouted_probs_home_first[tokens_to_move, home_idx] = 0

    return rerouting_home_first, rerouted_probs_home_first


def gen_offloading_plan_eager(
    map_token_to_expert: torch.Tensor,
    probs_routing: torch.Tensor,
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
    assignment_algorithm: AssignmentAlgorithm = "approx_bin_packing",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate an offloading plan with the retained Echo planner API."""

    _validate_common_inputs(
        map_token_to_expert,
        probs_routing,
        count_tokens_per_expert_from_ep_rank,
        num_ep_ranks,
        num_spare_experts_per_ep_rank,
    )

    if assignment_algorithm == "one_shot_greedy":
        count_tokens_from_home_expert_to_spare_expert, _, _ = gen_assignment(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            num_spare_experts_per_ep_rank,
            threshold_multiplier,
            dtype_index,
        )
    elif assignment_algorithm == "approx_bin_packing":
        if num_spare_experts_per_ep_rank != 1:
            raise ValueError(
                "approx_bin_packing only supports num_spare_experts_per_ep_rank=1, "
                f"got {num_spare_experts_per_ep_rank}"
            )
        count_tokens_from_home_expert_to_spare_expert, _, _ = gen_assignment_for_approx_bp(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            dtype_index,
        )
    else:
        raise ValueError(
            "Unknown assignment algorithm: "
            f"{assignment_algorithm}. Expected 'one_shot_greedy' or 'approx_bin_packing'."
        )

    count_tokens_from_home_expert_to_spare_expert = _select_unique_home_per_spare(
        count_tokens_from_home_expert_to_spare_expert
    )
    map_home_expert_to_spare = count_tokens_from_home_expert_to_spare_expert > 0
    first_pass, count_tokens_after_first_offload, capacity_spare_remaining = (
        breadth_first_allocation(
            count_tokens_per_expert_from_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
        )
    )
    second_pass, count_tokens_after_second_offload = depth_first_allocation(
        count_tokens_after_first_offload, capacity_spare_remaining
    )

    count_tokens_offloaded_to_spare = first_pass + second_pass
    count_tokens_offloaded_from_home = (
        count_tokens_per_expert_from_ep_rank - count_tokens_after_second_offload
    )
    rank = _ep_rank_to_int(ep_rank)
    rerouting_home_first, rerouted_probs_home_first = reroute_tokens_eager(
        map_token_to_expert,
        probs_routing,
        count_tokens_offloaded_from_home[rank].int(),
        count_tokens_offloaded_to_spare[rank].int(),
        map_home_expert_to_spare,
    )

    num_home_experts = map_token_to_expert.shape[1]
    num_spare_experts = num_spare_experts_per_ep_rank * num_ep_ranks
    rerouting_map = _to_effective_order(
        rerouting_home_first, num_home_experts, num_spare_experts, num_ep_ranks
    )
    rerouted_probs = _to_effective_order(
        rerouted_probs_home_first, num_home_experts, num_spare_experts, num_ep_ranks
    )

    return rerouting_map, rerouted_probs, map_home_expert_to_spare


_compiled_gen_offloading_plan = None


def _get_compiled_gen_offloading_plan():
    global _compiled_gen_offloading_plan
    if _compiled_gen_offloading_plan is None:
        _compiled_gen_offloading_plan = torch.compile(gen_offloading_plan_eager)
    return _compiled_gen_offloading_plan


def gen_offloading_plan(
    map_token_to_expert: torch.Tensor,
    probs_routing: torch.Tensor,
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
    assignment_algorithm: AssignmentAlgorithm = "approx_bin_packing",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production planner entrypoint.

    The compiled path is attempted by default, matching the old Echo production
    contract. Set ``MEGATRON_BALANCED_MOE_PLANNER_EAGER=1`` to force the eager
    fallback during local debugging.
    """

    if os.getenv("MEGATRON_BALANCED_MOE_PLANNER_EAGER") != "1":
        try:
            return _get_compiled_gen_offloading_plan()(
                map_token_to_expert,
                probs_routing,
                count_tokens_per_expert_from_ep_rank,
                ep_rank,
                num_ep_ranks,
                num_spare_experts_per_ep_rank,
                threshold_multiplier,
                dtype_index,
                assignment_algorithm,
            )
        except Exception:
            if os.getenv("MEGATRON_BALANCED_MOE_PLANNER_STRICT_COMPILE") == "1":
                raise

    return gen_offloading_plan_eager(
        map_token_to_expert,
        probs_routing,
        count_tokens_per_expert_from_ep_rank,
        ep_rank,
        num_ep_ranks,
        num_spare_experts_per_ep_rank,
        threshold_multiplier,
        dtype_index,
        assignment_algorithm,
    )


ep_group_random_generator = None
rank_random_generator = None


def generate_random_expert_offloading_map(
    num_home_experts: int,
    num_spare_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a deterministic random home-to-spare map for debug coverage."""

    global ep_group_random_generator
    if ep_group_random_generator is None:
        ep_group_random_generator = torch.Generator(device=device)
        ep_group_random_generator.manual_seed(42)

    selected_home_experts = torch.randint(
        0,
        num_home_experts,
        (num_spare_experts,),
        device=device,
        generator=ep_group_random_generator,
    )
    offloading_map = torch.zeros(
        num_home_experts, num_spare_experts, dtype=torch.bool, device=device
    )
    offloading_map[selected_home_experts, torch.arange(num_spare_experts, device=device)] = True
    return offloading_map


def gen_random_offloading_plan(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    ep: int,
    spare_expert_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a deterministic random offloading plan for debug and tests."""

    del tokens_per_expert_from_ep_rank, ep_rank, threshold_multiplier, index_dtype
    _validate_common_inputs(
        routing_map,
        probs,
        torch.zeros(ep, routing_map.shape[1], dtype=torch.int32, device=routing_map.device),
        ep,
        spare_expert_per_ep_rank,
    )

    global rank_random_generator
    device = routing_map.device
    if rank_random_generator is None:
        rank_random_generator = torch.Generator(device=device)
        rank_random_generator.manual_seed(42)

    num_tokens, num_home_experts = routing_map.shape
    num_spare_experts = spare_expert_per_ep_rank * ep
    expert_offloading_map = generate_random_expert_offloading_map(
        num_home_experts, num_spare_experts, device
    )

    rerouting_home_first = torch.zeros(
        num_tokens,
        num_home_experts + num_spare_experts,
        dtype=torch.bool,
        device=device,
    )
    rerouted_probs_home_first = torch.zeros(
        num_tokens,
        num_home_experts + num_spare_experts,
        dtype=probs.dtype,
        device=device,
    )
    rerouting_home_first[:, :num_home_experts] = routing_map
    rerouted_probs_home_first[:, :num_home_experts] = probs

    for home_expert_idx in range(num_home_experts):
        token_indices = torch.where(routing_map[:, home_expert_idx])[0]
        if token_indices.numel() == 0:
            continue

        spare_expert_indices = torch.where(expert_offloading_map[home_expert_idx])[0]
        if spare_expert_indices.numel() == 0:
            continue

        offload_ratio = (
            torch.rand(1, device=device, generator=rank_random_generator).item() * 0.2 + 0.2
        )
        num_tokens_to_offload = int(token_indices.numel() * offload_ratio)
        if num_tokens_to_offload == 0:
            continue

        perm = torch.randperm(
            token_indices.numel(), device=device, generator=rank_random_generator
        )[:num_tokens_to_offload]
        tokens_to_offload = token_indices[perm]
        spare_expert_idx = spare_expert_indices[
            torch.randint(
                spare_expert_indices.numel(),
                (1,),
                device=device,
                generator=rank_random_generator,
            ).item()
        ]
        spare_col = num_home_experts + int(spare_expert_idx.item())

        rerouting_home_first[tokens_to_offload, home_expert_idx] = False
        rerouting_home_first[tokens_to_offload, spare_col] = True
        rerouted_probs_home_first[tokens_to_offload, spare_col] = rerouted_probs_home_first[
            tokens_to_offload, home_expert_idx
        ]
        rerouted_probs_home_first[tokens_to_offload, home_expert_idx] = 0

    rerouting_map = _to_effective_order(
        rerouting_home_first, num_home_experts, num_spare_experts, ep
    )
    rerouted_probs = _to_effective_order(
        rerouted_probs_home_first, num_home_experts, num_spare_experts, ep
    )
    return rerouting_map, rerouted_probs, expert_offloading_map
