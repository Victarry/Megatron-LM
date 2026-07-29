# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import os

import pytest
import torch

_BENCHMARK_WARMUP_ITERS = 5
_BENCHMARK_ITERS = 20


def _require_offloading_planner():
    try:
        return importlib.import_module("megatron.core.transformer.moe.offloading_planner")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "megatron.core.transformer.moe.offloading_planner is required for "
            "BalancedMoELayer planner parity tests."
        )


def _cuda_event_latency_ms(fn, *, warmup_iters=_BENCHMARK_WARMUP_ITERS, iters=_BENCHMARK_ITERS):
    result = None
    for _ in range(warmup_iters):
        result = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, result


def _current_cuda_device():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    return torch.device("cuda", torch.cuda.current_device())


def _effective_column_for_home(home_idx, ep_size, home_per_rank, spare_per_rank):
    ep_rank = home_idx // home_per_rank
    local_home_idx = home_idx % home_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + local_home_idx


def _effective_column_for_spare(spare_idx, ep_size, home_per_rank, spare_per_rank):
    ep_rank = spare_idx // spare_per_rank
    local_spare_idx = spare_idx % spare_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + home_per_rank + local_spare_idx


def undo_reroute(
    rerouting_map: torch.Tensor,
    rerouted_probs: torch.Tensor,
    expert_offloading_map: torch.Tensor,
    *,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold spare columns back into their mapped home columns."""

    num_home_experts, num_spare_experts = expert_offloading_map.shape
    assert num_home_experts % ep_size == 0
    assert num_spare_experts % ep_size == 0
    home_per_rank = num_home_experts // ep_size
    spare_per_rank = num_spare_experts // ep_size

    home_map = torch.zeros(
        rerouting_map.shape[0],
        num_home_experts,
        dtype=rerouting_map.dtype,
        device=rerouting_map.device,
    )
    home_probs = torch.zeros(
        rerouted_probs.shape[0],
        num_home_experts,
        dtype=rerouted_probs.dtype,
        device=rerouted_probs.device,
    )

    for home_idx in range(num_home_experts):
        effective_col = _effective_column_for_home(home_idx, ep_size, home_per_rank, spare_per_rank)
        home_map[:, home_idx] |= rerouting_map[:, effective_col]
        home_probs[:, home_idx] += rerouted_probs[:, effective_col]

    home_indices, spare_indices = torch.where(expert_offloading_map)
    for home_idx, spare_idx in zip(home_indices.tolist(), spare_indices.tolist()):
        effective_col = _effective_column_for_spare(
            spare_idx, ep_size, home_per_rank, spare_per_rank
        )
        home_map[:, home_idx] |= rerouting_map[:, effective_col]
        home_probs[:, home_idx] += rerouted_probs[:, effective_col]

    return home_map, home_probs


def _make_local_routing(ep_size: int, topk: int, dtype: torch.dtype, device: torch.device):
    num_home_experts = 8
    tokens_per_rank = 12
    routing = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device
    )
    probs = torch.zeros(ep_size, tokens_per_rank, num_home_experts, dtype=dtype, device=device)

    for ep_rank in range(ep_size):
        for token_idx in range(tokens_per_rank):
            if token_idx < 8:
                first = 0
            else:
                first = (ep_rank * 2 + token_idx) % num_home_experts
            experts = [first]
            if topk == 2:
                second = (first + 3 + ep_rank) % num_home_experts
                if second == first:
                    second = (second + 1) % num_home_experts
                experts.append(second)
            values = [1.0] if topk == 1 else [0.625, 0.375]
            for expert_idx, prob in zip(experts, values):
                routing[ep_rank, token_idx, expert_idx] = True
                probs[ep_rank, token_idx, expert_idx] = prob

    return routing, probs


def _make_routing_from_counts(
    counts: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ep_size, num_home_experts = counts.shape
    tokens_per_rank = int(counts.sum(dim=1).max().item())
    routing = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device
    )
    probs = torch.zeros(ep_size, tokens_per_rank, num_home_experts, dtype=dtype, device=device)

    for ep_rank in range(ep_size):
        offset = 0
        for expert_idx, count in enumerate(counts[ep_rank].tolist()):
            end = offset + int(count)
            if end > offset:
                routing[ep_rank, offset:end, expert_idx] = True
                probs[ep_rank, offset:end, expert_idx] = 1.0
            offset = end

    return routing, probs


def _make_qwen3_30b_routing(dtype: torch.dtype, device: torch.device):
    ep_size = 8
    num_home_experts = 128
    topk = 8
    tokens_per_rank = 32
    routing = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device
    )
    probs = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=dtype, device=device
    )
    prob_values = torch.arange(topk, 0, -1, dtype=torch.float32, device=device)
    prob_values = (prob_values / prob_values.sum()).to(dtype)

    for ep_rank in range(ep_size):
        for token_idx in range(tokens_per_rank):
            if token_idx < 24:
                experts = list(range(topk))
            else:
                start = (ep_rank * 17 + token_idx * topk) % (num_home_experts - topk)
                experts = [
                    topk + (start + offset) % (num_home_experts - topk)
                    for offset in range(topk)
                ]
            routing[ep_rank, token_idx, experts] = True
            probs[ep_rank, token_idx, experts] = prob_values
    return routing, probs


def _home_only_effective(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    *,
    ep_size: int,
    num_spare_experts: int,
):
    num_home_experts = routing_map.shape[1]
    home_per_rank = num_home_experts // ep_size
    spare_per_rank = num_spare_experts // ep_size
    effective_map = torch.zeros(
        routing_map.shape[0],
        num_home_experts + num_spare_experts,
        dtype=torch.bool,
        device=routing_map.device,
    )
    effective_probs = torch.zeros(
        probs.shape[0],
        num_home_experts + num_spare_experts,
        dtype=probs.dtype,
        device=probs.device,
    )
    for home_idx in range(num_home_experts):
        effective_col = _effective_column_for_home(
            home_idx, ep_size, home_per_rank, spare_per_rank
        )
        effective_map[:, effective_col] = routing_map[:, home_idx]
        effective_probs[:, effective_col] = probs[:, home_idx]
    return effective_map, effective_probs


def _spare_columns(num_home_experts, num_spare_experts, ep_size):
    home_per_rank = num_home_experts // ep_size
    spare_per_rank = num_spare_experts // ep_size
    return [
        _effective_column_for_spare(spare_idx, ep_size, home_per_rank, spare_per_rank)
        for spare_idx in range(num_spare_experts)
    ]


def _moved_assignment_count(rerouting_map, expert_offloading_map, ep_size):
    spare_columns = _spare_columns(
        expert_offloading_map.shape[0], expert_offloading_map.shape[1], ep_size
    )
    return int(rerouting_map[:, spare_columns].sum().item())


def _effective_rank_loads(rerouting_maps, ep_size, num_spare_experts):
    effective = torch.cat(rerouting_maps, dim=0)
    effective_per_rank = effective.shape[1] // ep_size
    assert effective_per_rank * ep_size == effective.shape[1]
    assert num_spare_experts % ep_size == 0
    return effective.sum(dim=0).reshape(ep_size, effective_per_rank).sum(dim=1)


def _assert_plans_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _reset_random_generators(planner):
    planner.ep_group_random_generator = None
    planner.rank_random_generator = None


def _assert_equivalent_plan(
    original_map, original_probs, rerouting_map, rerouted_probs, expert_offloading_map, ep_size
):
    num_home_experts, num_spare_experts = expert_offloading_map.shape
    expected_shape = (original_map.shape[0], num_home_experts + num_spare_experts)
    assert rerouting_map.shape == expected_shape
    assert rerouted_probs.shape == expected_shape
    assert rerouting_map.dtype is torch.bool
    assert rerouted_probs.dtype is original_probs.dtype
    assert expert_offloading_map.dtype is torch.bool
    assert rerouting_map.device == original_map.device
    assert rerouted_probs.device == original_probs.device
    assert expert_offloading_map.device == original_map.device

    restored_map, restored_probs = undo_reroute(
        rerouting_map, rerouted_probs, expert_offloading_map, ep_size=ep_size
    )
    assert torch.equal(restored_map, original_map)
    torch.testing.assert_close(restored_probs, original_probs, rtol=0, atol=0)
    assert expert_offloading_map.sum(dim=0).max().item() <= 1
    assert torch.equal(rerouting_map.sum(dim=1), original_map.sum(dim=1))
    torch.testing.assert_close(rerouted_probs.sum(dim=1), original_probs.sum(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(
        rerouted_probs.masked_select(~rerouting_map),
        torch.zeros_like(rerouted_probs.masked_select(~rerouting_map)),
        rtol=0,
        atol=0,
    )

    home_per_rank = num_home_experts // ep_size
    spare_per_rank = num_spare_experts // ep_size
    for spare_idx in range(num_spare_experts):
        if expert_offloading_map[:, spare_idx].any():
            continue
        effective_col = _effective_column_for_spare(
            spare_idx, ep_size, home_per_rank, spare_per_rank
        )
        assert not rerouting_map[:, effective_col].any()
        torch.testing.assert_close(
            rerouted_probs[:, effective_col],
            torch.zeros_like(rerouted_probs[:, effective_col]),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
@pytest.mark.parametrize(
    "topk,assignment_algorithm,spare_per_ep",
    [(1, "approx_bin_packing", 1), (2, "one_shot_greedy", 2)],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_offloading_planner_equivalence(topk, assignment_algorithm, spare_per_ep, dtype):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk, dtype, device)
    counts = routing.sum(dim=1).to(torch.int32)

    rerouting_by_rank = []
    offloading_maps = []
    for ep_rank in range(ep_size):
        rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_offloading_plan(
            routing[ep_rank],
            probs[ep_rank],
            counts,
            ep_rank,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=spare_per_ep,
            assignment_algorithm=assignment_algorithm,
        )
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )
        rerouting_by_rank.append(rerouting_map)
        offloading_maps.append(expert_offloading_map)

    assert all(torch.equal(offloading_maps[0], m) for m in offloading_maps[1:])

    before_rank_load = (
        routing.reshape(-1, routing.shape[-1]).sum(dim=0).reshape(ep_size, -1).sum(dim=1)
    )
    after_rank_load = _effective_rank_loads(
        rerouting_by_rank, ep_size, num_spare_experts=spare_per_ep * ep_size
    )
    assert offloading_maps[0].any()
    assert sum(
        _moved_assignment_count(rerouting, offloading_maps[0], ep_size)
        for rerouting in rerouting_by_rank
    ) > 0
    assert after_rank_load.max() < before_rank_load.max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_balanced_input_is_exact_noop():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    num_home_experts = 8
    spare_per_ep = 1
    counts = torch.ones(ep_size, num_home_experts, dtype=torch.int32, device=device)
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    for ep_rank in range(ep_size):
        actual = planner.gen_offloading_plan_eager(
            routing[ep_rank],
            probs[ep_rank],
            counts,
            ep_rank,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=spare_per_ep,
            assignment_algorithm="approx_bin_packing",
        )
        rerouting_map, rerouted_probs, expert_offloading_map = actual
        expected_map, expected_probs = _home_only_effective(
            routing[ep_rank],
            probs[ep_rank],
            ep_size=ep_size,
            num_spare_experts=spare_per_ep * ep_size,
        )
        torch.testing.assert_close(rerouting_map, expected_map, rtol=0, atol=0)
        torch.testing.assert_close(rerouted_probs, expected_probs, rtol=0, atol=0)
        assert not expert_offloading_map.any()
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_one_shot_greedy_matches_hand_computed_multi_spare_plan():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    num_home_experts = 8
    spare_per_ep = 2
    counts = torch.zeros(ep_size, num_home_experts, dtype=torch.int32, device=device)
    counts[0, [0, 2, 4, 6]] = torch.tensor(
        [100, 20, 25, 15], dtype=torch.int32, device=device
    )
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    rerouting_by_rank = []
    offloading_maps = []
    for ep_rank in range(ep_size):
        rerouting_map, rerouted_probs, expert_offloading_map = (
            planner.gen_offloading_plan_eager(
                routing[ep_rank],
                probs[ep_rank],
                counts,
                ep_rank,
                num_ep_ranks=ep_size,
                num_spare_experts_per_ep_rank=spare_per_ep,
                assignment_algorithm="one_shot_greedy",
            )
        )
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )
        expected_moved = 60 if ep_rank == 0 else 0
        assert (
            _moved_assignment_count(rerouting_map, expert_offloading_map, ep_size)
            == expected_moved
        )
        rerouting_by_rank.append(rerouting_map)
        offloading_maps.append(expert_offloading_map)

    assert all(torch.equal(offloading_maps[0], value) for value in offloading_maps[1:])
    expert_offloading_map = offloading_maps[0]
    assert expert_offloading_map.sum().item() == 2
    assert expert_offloading_map[0, 2]
    assert expert_offloading_map[0, 6]
    assert not expert_offloading_map[:, :2].any()
    assert not expert_offloading_map[:, 3:6].any()
    assert not expert_offloading_map[:, 7:].any()
    torch.testing.assert_close(
        _effective_rank_loads(
            rerouting_by_rank,
            ep_size,
            num_spare_experts=ep_size * spare_per_ep,
        ),
        torch.tensor([40, 30, 25, 65], dtype=torch.int64, device=device),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_qwen3_30b_top8_production_shape():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 8
    spare_per_ep = 1
    routing, probs = _make_qwen3_30b_routing(torch.bfloat16, device)
    counts = routing.sum(dim=1).to(torch.int32)

    rerouting_by_rank = []
    offloading_maps = []
    for ep_rank in range(ep_size):
        rerouting_map, rerouted_probs, expert_offloading_map = (
            planner.gen_offloading_plan_eager(
                routing[ep_rank],
                probs[ep_rank],
                counts,
                ep_rank,
                num_ep_ranks=ep_size,
                num_spare_experts_per_ep_rank=spare_per_ep,
                assignment_algorithm="approx_bin_packing",
            )
        )
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )
        rerouting_by_rank.append(rerouting_map)
        offloading_maps.append(expert_offloading_map)

    assert all(torch.equal(offloading_maps[0], value) for value in offloading_maps[1:])
    assert offloading_maps[0].any()
    assert sum(
        _moved_assignment_count(rerouting, offloading_maps[0], ep_size)
        for rerouting in rerouting_by_rank
    ) > 0
    before_rank_load = (
        routing.reshape(-1, routing.shape[-1]).sum(dim=0).reshape(ep_size, -1).sum(dim=1)
    )
    after_rank_load = _effective_rank_loads(
        rerouting_by_rank, ep_size, num_spare_experts=ep_size * spare_per_ep
    )
    assert after_rank_load.max() < before_rank_load.max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_handles_uneven_sources_zero_token_rank_and_padding():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    num_home_experts = 8
    spare_per_ep = 2
    counts = torch.zeros(ep_size, num_home_experts, dtype=torch.int32, device=device)
    counts[:, 0] = torch.tensor([12, 3, 0, 1], dtype=torch.int32, device=device)
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    rerouting_by_rank = []
    offloading_maps = []
    for ep_rank in range(ep_size):
        rerouting_map, rerouted_probs, expert_offloading_map = (
            planner.gen_offloading_plan_eager(
                routing[ep_rank],
                probs[ep_rank],
                counts,
                ep_rank,
                num_ep_ranks=ep_size,
                num_spare_experts_per_ep_rank=spare_per_ep,
                assignment_algorithm="one_shot_greedy",
            )
        )
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )
        zero_route_rows = ~routing[ep_rank].any(dim=1)
        assert not rerouting_map[zero_route_rows].any()
        torch.testing.assert_close(
            rerouted_probs[zero_route_rows],
            torch.zeros_like(rerouted_probs[zero_route_rows]),
            rtol=0,
            atol=0,
        )
        rerouting_by_rank.append(rerouting_map)
        offloading_maps.append(expert_offloading_map)

    assert all(torch.equal(offloading_maps[0], value) for value in offloading_maps[1:])
    assert sum(
        _moved_assignment_count(rerouting, offloading_maps[0], ep_size)
        for rerouting in rerouting_by_rank
    ) > 0
    assert _moved_assignment_count(rerouting_by_rank[2], offloading_maps[0], ep_size) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_approx_bin_packing_keeps_one_home_per_spare_slot():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 8
    num_home_experts = 32
    counts = torch.zeros(ep_size, num_home_experts, dtype=torch.int32, device=device)
    counts[0, 0:4] = torch.tensor([80, 90, 110, 120], dtype=torch.int32, device=device)
    counts[0, 8:32] = 50
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_offloading_plan(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        num_ep_ranks=ep_size,
        num_spare_experts_per_ep_rank=1,
        assignment_algorithm="approx_bin_packing",
    )

    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )
    assert expert_offloading_map.any(dim=0).sum().item() == 1
    assert expert_offloading_map[:, 1].sum().item() == 1
    assert expert_offloading_map[3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_random_offloading_equivalence():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    topk = 2
    routing, probs = _make_local_routing(ep_size, topk, torch.float32, device)
    counts = routing.sum(dim=1).to(torch.int32)

    rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_random_offloading_plan(
        routing[0], probs[0], counts, ep_rank=0, ep=ep_size, spare_expert_per_ep_rank=1
    )
    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_random_offloading_is_cross_rank_reproducible_and_moves_multi_spare_tokens():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    num_home_experts = 8
    spare_per_ep = 2
    tokens_per_rank = 64
    routing = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device
    )
    probs = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.float32, device=device
    )
    for ep_rank in range(ep_size):
        for token_idx in range(tokens_per_rank):
            first = (token_idx + ep_rank) % num_home_experts
            second = (first + 3) % num_home_experts
            routing[ep_rank, token_idx, first] = True
            routing[ep_rank, token_idx, second] = True
            probs[ep_rank, token_idx, first] = 0.625
            probs[ep_rank, token_idx, second] = 0.375
    counts = routing.sum(dim=1).to(torch.int32)

    plans = []
    for ep_rank in range(ep_size):
        _reset_random_generators(planner)
        plan = planner.gen_random_offloading_plan(
            routing[ep_rank],
            probs[ep_rank],
            counts,
            ep_rank=ep_rank,
            ep=ep_size,
            spare_expert_per_ep_rank=spare_per_ep,
        )
        _assert_equivalent_plan(routing[ep_rank], probs[ep_rank], *plan, ep_size)
        assert _moved_assignment_count(plan[0], plan[2], ep_size) > 0
        plans.append(plan)

    assert all(torch.equal(plans[0][2], plan[2]) for plan in plans[1:])
    assert plans[0][2].sum(dim=0).eq(1).all()

    _reset_random_generators(planner)
    first_replay = planner.gen_random_offloading_plan(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        ep=ep_size,
        spare_expert_per_ep_rank=spare_per_ep,
    )
    _reset_random_generators(planner)
    second_replay = planner.gen_random_offloading_plan(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        ep=ep_size,
        spare_expert_per_ep_rank=spare_per_ep,
    )
    _assert_plans_equal(first_replay, second_replay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
@pytest.mark.parametrize("assignment_algorithm", ["one_shot_greedy", "approx_bin_packing"])
@pytest.mark.parametrize(
    "threshold_multiplier,expect_movement",
    [(0.0, True), (0.09, True), (0.10, False), (1.0, False)],
)
def test_offloading_planner_threshold_reclaims_unnecessary_assignments(
    assignment_algorithm, threshold_multiplier, expect_movement
):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    num_home_experts = 8
    global_counts = torch.tensor(
        [11, 11, 10, 10, 9, 9, 10, 10], dtype=torch.int32, device=device
    )
    counts = torch.zeros(ep_size, num_home_experts, dtype=torch.int32, device=device)
    counts[0] = global_counts
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_offloading_plan_eager(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        num_ep_ranks=ep_size,
        num_spare_experts_per_ep_rank=1,
        threshold_multiplier=threshold_multiplier,
        assignment_algorithm=assignment_algorithm,
    )
    _assert_equivalent_plan(
        routing[0],
        probs[0],
        rerouting_map,
        rerouted_probs,
        expert_offloading_map,
        ep_size,
    )
    assert bool(expert_offloading_map.any().item()) == expect_movement
    assert (
        _moved_assignment_count(rerouting_map, expert_offloading_map, ep_size) > 0
    ) is expect_movement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
@pytest.mark.parametrize(
    "assignment_algorithm,spare_per_ep",
    [("approx_bin_packing", 1), ("one_shot_greedy", 2)],
)
def test_compiled_offloading_planner_matches_eager_strictly(
    monkeypatch, assignment_algorithm, spare_per_ep
):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk=2, dtype=torch.float32, device=device)
    counts = routing.sum(dim=1).to(torch.int32)
    args = (
        routing[0],
        probs[0],
        counts,
        0,
        ep_size,
        spare_per_ep,
        0.0,
        torch.int32,
        assignment_algorithm,
    )
    expected = planner.gen_offloading_plan_eager(*args)

    monkeypatch.delenv("MEGATRON_BALANCED_MOE_PLANNER_EAGER", raising=False)
    monkeypatch.setenv("MEGATRON_BALANCED_MOE_PLANNER_STRICT_COMPILE", "1")
    monkeypatch.setattr(planner, "_compiled_gen_offloading_plan", None)
    actual = planner.gen_offloading_plan(*args)

    _assert_plans_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_compile_failure_falls_back_unless_strict(monkeypatch):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk=1, dtype=torch.float32, device=device)
    counts = routing.sum(dim=1).to(torch.int32)
    args = (routing[0], probs[0], counts, 0, ep_size, 1)
    expected = planner.gen_offloading_plan_eager(*args)
    compile_calls = {"count": 0}

    def broken_compiled_planner(*args, **kwargs):
        compile_calls["count"] += 1
        raise RuntimeError("simulated planner compile failure")

    monkeypatch.delenv("MEGATRON_BALANCED_MOE_PLANNER_EAGER", raising=False)
    monkeypatch.delenv("MEGATRON_BALANCED_MOE_PLANNER_STRICT_COMPILE", raising=False)
    monkeypatch.setattr(planner, "_compiled_gen_offloading_plan", broken_compiled_planner)
    actual = planner.gen_offloading_plan(*args)
    _assert_plans_equal(actual, expected)
    assert compile_calls["count"] == 1

    monkeypatch.setenv("MEGATRON_BALANCED_MOE_PLANNER_STRICT_COMPILE", "1")
    with pytest.raises(RuntimeError, match="simulated planner compile failure"):
        planner.gen_offloading_plan(*args)
    assert compile_calls["count"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
@pytest.mark.parametrize(
    "case,error_match",
    [
        ("routing_ndim", "2D routing map"),
        ("routing_dtype", "bool tensor"),
        ("prob_shape", "same shape"),
        ("prob_dtype", "floating-point"),
        ("count_ndim", "counts shaped"),
        ("count_dtype", "integer token counts"),
        ("negative_count", "non-negative token counts"),
        ("count_ep_shape", "EP dimension"),
        ("count_expert_shape", "expert dimension"),
        ("nondivisible_experts", "experts divisible"),
        ("nonpositive_ep", "positive num_ep_ranks"),
        ("nonpositive_spares", "positive spare expert count"),
        ("negative_ep_rank", "ep_rank"),
        ("high_ep_rank", "ep_rank"),
        ("negative_threshold", "non-negative threshold"),
        ("nan_threshold", "finite threshold"),
        ("unknown_algorithm", "Unknown assignment algorithm"),
    ],
)
def test_offloading_planner_rejects_invalid_public_inputs(case, error_match):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    all_routing, all_probs = _make_local_routing(
        ep_size, topk=1, dtype=torch.float32, device=device
    )
    routing = all_routing[0]
    probs = all_probs[0]
    counts = all_routing.sum(dim=1).to(torch.int32)
    ep_rank = 0
    num_ep_ranks = ep_size
    spare_per_ep = 1
    threshold_multiplier = 0.0
    assignment_algorithm = "approx_bin_packing"

    if case == "routing_ndim":
        routing = routing.unsqueeze(0)
        probs = probs.unsqueeze(0)
    elif case == "routing_dtype":
        routing = routing.to(torch.int32)
    elif case == "prob_shape":
        probs = probs[:, :-1]
    elif case == "prob_dtype":
        probs = probs.to(torch.int32)
    elif case == "count_ndim":
        counts = counts.unsqueeze(0)
    elif case == "count_dtype":
        counts = counts.to(torch.float32)
    elif case == "negative_count":
        counts = counts.clone()
        counts[0, 0] = -1
    elif case == "count_ep_shape":
        counts = counts[:-1]
    elif case == "count_expert_shape":
        counts = counts[:, :-1]
    elif case == "nondivisible_experts":
        routing = torch.zeros(4, 10, dtype=torch.bool, device=device)
        probs = torch.zeros(4, 10, dtype=torch.float32, device=device)
        counts = torch.zeros(ep_size, 10, dtype=torch.int32, device=device)
    elif case == "nonpositive_ep":
        num_ep_ranks = 0
    elif case == "nonpositive_spares":
        spare_per_ep = 0
    elif case == "negative_ep_rank":
        ep_rank = -1
    elif case == "high_ep_rank":
        ep_rank = ep_size
    elif case == "negative_threshold":
        threshold_multiplier = -0.1
    elif case == "nan_threshold":
        threshold_multiplier = float("nan")
    elif case == "unknown_algorithm":
        assignment_algorithm = "unknown"
    else:
        raise AssertionError(f"Unhandled validation case: {case}")

    with pytest.raises(ValueError, match=error_match):
        planner.gen_offloading_plan(
            routing,
            probs,
            counts,
            ep_rank,
            num_ep_ranks=num_ep_ranks,
            num_spare_experts_per_ep_rank=spare_per_ep,
            threshold_multiplier=threshold_multiplier,
            assignment_algorithm=assignment_algorithm,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_approx_bin_packing_rejects_multiple_spares_per_ep():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk=1, dtype=torch.float32, device=device)
    counts = routing.sum(dim=1).to(torch.int32)

    with pytest.raises(ValueError, match="approx_bin_packing"):
        planner.gen_offloading_plan(
            routing[0],
            probs[0],
            counts,
            ep_rank=0,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=2,
            assignment_algorithm="approx_bin_packing",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_kernel_latency_benchmark():
    if os.getenv("MEGATRON_BALANCED_MOE_PLANNER_EAGER") == "1":
        pytest.skip("Planner latency benchmark targets the production compiled path.")

    planner = _require_offloading_planner()
    device = _current_cuda_device()
    ep_size = 8
    num_home_experts = 32
    counts = torch.full((ep_size, num_home_experts), 8, dtype=torch.int32, device=device)
    counts[0, 0:4] = torch.tensor([512, 480, 448, 416], dtype=torch.int32, device=device)
    counts[0, 4:] = 16
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    def run_planner():
        return planner.gen_offloading_plan(
            routing[0],
            probs[0],
            counts,
            ep_rank=0,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=1,
            assignment_algorithm="approx_bin_packing",
        )

    latency_ms, (rerouting_map, rerouted_probs, expert_offloading_map) = _cuda_event_latency_ms(
        run_planner
    )
    assert latency_ms > 0.0
    assert torch.isfinite(torch.tensor(latency_ms, device=device))
    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )
    print(
        "BENCHMARK balanced_moe_planner_kernel "
        f"algorithm=approx_bin_packing ep_size={ep_size} "
        f"num_home_experts={num_home_experts} local_tokens={routing.shape[1]} "
        f"warmup_iters={_BENCHMARK_WARMUP_ITERS} iters={_BENCHMARK_ITERS} "
        f"mean_ms={latency_ms:.4f}",
        flush=True,
    )
