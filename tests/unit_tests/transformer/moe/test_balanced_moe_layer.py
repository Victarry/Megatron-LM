# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import os

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _require_distributed_cuda(world_size=4):
    if not torch.cuda.is_available():
        pytest.skip("BalancedMoELayer parity tests require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) < world_size:
        pytest.skip(f"Run with torchrun --nproc-per-node={world_size}.")


def _require_balanced_moe_layer():
    try:
        module = importlib.import_module("megatron.core.transformer.moe.balanced_moe_layer")
    except ModuleNotFoundError:
        pytest.fail(
            "megatron.core.transformer.moe.balanced_moe_layer is required for "
            "BalancedMoELayer parity tests."
        )
    try:
        return module, module.BalancedMoELayer
    except AttributeError:
        pytest.fail("BalancedMoELayer class is required in balanced_moe_layer.py.")


def _require_symmetric_memory_backend():
    dispatcher_module = importlib.import_module(
        "megatron.core.transformer.moe.expert_weight_dispatcher"
    )
    dispatcher_cls = getattr(dispatcher_module, "SymmetricMemoryExpertWeightDispatcher", None)
    if dispatcher_cls is None:
        pytest.fail("SymmetricMemoryExpertWeightDispatcher is required.")
    availability_error = dispatcher_cls.availability_error()
    if availability_error is not None:
        pytest.skip(f"Symmetric Memory expert dispatch is unavailable: {availability_error}")
    return dispatcher_cls


def _require_hybridep_backend():
    dispatcher_module = importlib.import_module(
        "megatron.core.transformer.moe.expert_weight_dispatcher"
    )
    dispatcher_cls = getattr(dispatcher_module, "HybridEPExpertWeightDispatcher", None)
    if dispatcher_cls is None:
        pytest.fail("HybridEPExpertWeightDispatcher is required.")
    availability_error = dispatcher_cls.availability_error()
    if availability_error is not None:
        pytest.skip(f"HybridEP expert dispatch is unavailable: {availability_error}")
    return dispatcher_cls


def _make_config(
    dtype,
    topk,
    *,
    balanced,
    random_offloading=False,
    grouped=False,
    shared_expert=False,
    recompute_expert_dispatch=False,
    expert_weight_backend="all_to_all",
    grad_combine_dtype="fp32",
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=32,
        moe_router_topk=topk,
        moe_router_pre_softmax=topk == 1,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped,
        expert_model_parallel_size=4,
        tensor_model_parallel_size=1,
        add_bias_linear=False,
        params_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )
    if shared_expert:
        config.moe_shared_expert_intermediate_size = 32
        config.moe_shared_expert_overlap = False
    if balanced:
        config.moe_use_balanced_layer = True
        config.moe_num_spare_experts = 4
        config.moe_balance_assignment_algorithm = "approx_bin_packing"
        config.moe_balance_enable_random_offloading = random_offloading
        config.moe_balance_threshold_multiplier = 0.0
        config.moe_balance_debug_dump_path = None
        config.moe_balance_enable_debug_stats = True
        config.moe_balance_recompute_expert_dispatch = recompute_expert_dispatch
        config.moe_balance_expert_weight_dispatch_backend = expert_weight_backend
        config.moe_balance_expert_weight_grad_combine_dtype = grad_combine_dtype
    return config


def _balanced_config_kwargs(**overrides):
    kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 32,
        "moe_router_topk": 1,
        "moe_router_pre_softmax": True,
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_token_dispatcher_type": "alltoall",
        "moe_grouped_gemm": False,
        "expert_model_parallel_size": 4,
        "tensor_model_parallel_size": 1,
        "add_bias_linear": False,
        "params_dtype": torch.float32,
        "moe_use_balanced_layer": True,
        "moe_num_spare_experts": 4,
        "moe_balance_assignment_algorithm": "approx_bin_packing",
        "moe_balance_expert_weight_dispatch_backend": "all_to_all",
        "moe_balance_expert_weight_grad_combine_dtype": "fp32",
    }
    kwargs.update(overrides)
    return kwargs


def _submodules(grouped=False):
    if grouped:
        spec = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=8, moe_grouped_gemm=True
        ).mlp
    else:
        spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
    submodules = get_submodules(spec)
    assert isinstance(submodules, MoESubmodules)
    return submodules


def _copy_home_state(baseline: MoELayer, balanced):
    balanced.router.load_state_dict(baseline.router.state_dict())
    if hasattr(baseline.experts, "local_experts"):
        for local_idx, baseline_expert in enumerate(baseline.experts.local_experts):
            balanced_expert = balanced.experts.local_experts[local_idx]
            balanced_expert.load_state_dict(baseline_expert.state_dict())
    else:
        for module_name in ("linear_fc1", "linear_fc2"):
            baseline_layer = getattr(baseline.experts, module_name)
            balanced_layer = getattr(balanced.experts, module_name)
            for local_idx in range(balanced.num_local_home_experts):
                for prefix in ("weight", "bias"):
                    name = f"{prefix}{local_idx}"
                    if hasattr(baseline_layer, name):
                        getattr(balanced_layer, name).data.copy_(getattr(baseline_layer, name).data)

    if baseline.shared_experts is not None:
        assert balanced.shared_experts is not None
        balanced.shared_experts.load_state_dict(baseline.shared_experts.state_dict())


def _effective_col_home(home_idx, ep_size=4, home_per_rank=2, spare_per_rank=1):
    ep_rank = home_idx // home_per_rank
    local_home = home_idx % home_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + local_home


def _effective_col_spare(spare_idx, ep_size=4, home_per_rank=2, spare_per_rank=1):
    ep_rank = spare_idx // spare_per_rank
    local_spare = spare_idx % spare_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + home_per_rank + local_spare


def _extend_home_routing(routing_map, probs):
    num_effective_experts = 12
    extended_map = torch.zeros(
        routing_map.shape[0], num_effective_experts, dtype=torch.bool, device=routing_map.device
    )
    extended_probs = torch.zeros(
        routing_map.shape[0], num_effective_experts, dtype=probs.dtype, device=probs.device
    )
    for home_idx in range(8):
        effective_col = _effective_col_home(home_idx)
        extended_map[:, effective_col] = routing_map[:, home_idx]
        extended_probs[:, effective_col] = probs[:, home_idx]
    return (
        extended_map,
        extended_probs,
        torch.zeros(8, 4, dtype=torch.bool, device=routing_map.device),
    )


def _forced_move_plan(routing_map, probs):
    extended_map, extended_probs, expert_map = _extend_home_routing(routing_map, probs)
    active_tokens = torch.where(routing_map[:, 0])[0]
    if active_tokens.numel() == 0:
        active_tokens = torch.where(routing_map.any(dim=1))[0]
        home_idx = torch.where(routing_map[active_tokens[0]])[0][0].item()
    else:
        home_idx = 0
    moved_token = active_tokens[0].item()
    spare_idx = 0
    expert_map[home_idx, spare_idx] = True
    home_col = _effective_col_home(home_idx)
    spare_col = _effective_col_spare(spare_idx)
    extended_map[moved_token, home_col] = False
    extended_map[moved_token, spare_col] = True
    extended_probs[moved_token, spare_col] = extended_probs[moved_token, home_col]
    extended_probs[moved_token, home_col] = 0
    return extended_map, extended_probs, expert_map


def _patch_planner(monkeypatch, balanced_module, mode):
    def plan(routing_map, probs, *args, **kwargs):
        if mode == "no_move":
            return _extend_home_routing(routing_map, probs)
        if mode in {"forced_move", "random"}:
            return _forced_move_plan(routing_map, probs)
        raise AssertionError(f"unknown planner mode {mode}")

    monkeypatch.setattr(balanced_module, "gen_offloading_plan", plan, raising=False)
    monkeypatch.setattr(balanced_module, "gen_random_offloading_plan", plan, raising=False)
    try:
        planner_module = importlib.import_module("megatron.core.transformer.moe.offloading_planner")
    except ModuleNotFoundError:
        return
    monkeypatch.setattr(planner_module, "gen_offloading_plan", plan, raising=False)
    monkeypatch.setattr(planner_module, "gen_random_offloading_plan", plan, raising=False)


def _run_layer(layer, hidden_states, output_grad, **forward_kwargs):
    output, _ = layer(hidden_states, **forward_kwargs)
    output.backward(output_grad)
    return output.detach()


def _assert_router_grads_close(baseline, balanced):
    for baseline_param, balanced_param in zip(
        baseline.router.parameters(), balanced.router.parameters()
    ):
        _assert_grad_close(balanced_param, baseline_param)


def _assert_grad_close(actual_param, expected_param, *, rtol=None, atol=None):
    actual_grad = getattr(actual_param, "main_grad", None)
    expected_grad = getattr(expected_param, "main_grad", None)
    if actual_grad is None:
        actual_grad = actual_param.grad
    if expected_grad is None:
        expected_grad = expected_param.grad
    if expected_grad is None or actual_grad is None:
        assert actual_grad is None
        assert expected_grad is None
        return
    assert_close_kwargs = {}
    if rtol is not None:
        assert_close_kwargs["rtol"] = rtol
    if atol is not None:
        assert_close_kwargs["atol"] = atol
    if actual_grad.dtype != expected_grad.dtype:
        torch.testing.assert_close(
            actual_grad.to(dtype=expected_grad.dtype), expected_grad, **assert_close_kwargs
        )
    else:
        torch.testing.assert_close(actual_grad, expected_grad, **assert_close_kwargs)


def _assert_home_expert_grads_close(baseline, balanced, *, rtol=None, atol=None):
    if hasattr(baseline.experts, "local_experts"):
        for local_idx, baseline_expert in enumerate(baseline.experts.local_experts):
            balanced_expert = balanced.experts.local_experts[local_idx]
            for baseline_param, balanced_param in zip(
                baseline_expert.parameters(), balanced_expert.parameters()
            ):
                _assert_grad_close(balanced_param, baseline_param, rtol=rtol, atol=atol)
        return

    for module_name in ("linear_fc1", "linear_fc2"):
        baseline_layer = getattr(baseline.experts, module_name)
        balanced_layer = getattr(balanced.experts, module_name)
        for local_idx in range(balanced.num_local_home_experts):
            for prefix in ("weight", "bias"):
                name = f"{prefix}{local_idx}"
                if hasattr(baseline_layer, name):
                    _assert_grad_close(
                        getattr(balanced_layer, name),
                        getattr(baseline_layer, name),
                        rtol=rtol,
                        atol=atol,
                    )


def _assert_shared_expert_grads_close(baseline, balanced):
    if baseline.shared_experts is None:
        assert balanced.shared_experts is None
        return
    assert balanced.shared_experts is not None
    for baseline_param, balanced_param in zip(
        baseline.shared_experts.parameters(), balanced.shared_experts.parameters()
    ):
        _assert_grad_close(balanced_param, baseline_param)


def _assert_no_spare_parameters(balanced):
    names = [name for name, _ in balanced.named_parameters()]
    for spare_idx in balanced.local_spare_expert_indices:
        assert not any(f"local_experts.{spare_idx}." in name for name in names)
        assert not any(f"linear_fc1.weight{spare_idx}" in name for name in names)
        assert not any(f"linear_fc2.weight{spare_idx}" in name for name in names)


@pytest.mark.internal
@pytest.mark.parametrize(
    "mode,topk,dtype",
    [
        ("no_move", 1, torch.float64),
        ("no_move", 2, torch.float64),
        ("forced_move", 1, torch.float64),
        ("forced_move", 2, torch.float64),
        ("random", 1, torch.float32),
        ("planner", 1, torch.float32),
        ("planner", 1, torch.bfloat16),
    ],
)
def test_balanced_moe_layer_parity(monkeypatch, mode, topk, dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        if mode != "planner":
            _patch_planner(monkeypatch, balanced_module, mode)

        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(
            _make_config(dtype, topk, balanced=False), _submodules(), layer_number=1
        )
        balanced_config = _make_config(
            dtype, topk, balanced=True, random_offloading=(mode == "random")
        )
        balanced = balanced_cls(balanced_config, _submodules(), layer_number=1)
        baseline.cuda().to(dtype=dtype)
        balanced.cuda().to(dtype=dtype)
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)

        baseline_output = _run_layer(baseline, hidden, output_grad)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        _assert_home_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)

        if mode != "no_move":
            assert balanced.last_debug_stats is not None
            assert balanced.last_debug_stats.num_global_active_spare_slots > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("expert_weight_backend", ["all_to_all", "symmetric_memory", "hybridep"])
def test_balanced_moe_layer_recomputes_expert_dispatch(monkeypatch, expert_weight_backend):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        _patch_planner(monkeypatch, balanced_module, "forced_move")

        dispatcher_module = importlib.import_module(
            "megatron.core.transformer.moe.expert_weight_dispatcher"
        )
        if expert_weight_backend == "symmetric_memory":
            dispatcher_cls = _require_symmetric_memory_backend()
        elif expert_weight_backend == "hybridep":
            dispatcher_cls = _require_hybridep_backend()
        else:
            dispatcher_cls = dispatcher_module.AllToAllExpertWeightDispatcher
        original_dispatch = dispatcher_cls.dispatch
        dispatch_calls = {"no_grad": 0, "grad": 0}

        def counted_dispatch(self, metadata, *expert_weights, runtime_weight_dtype=None):
            if torch.is_grad_enabled():
                dispatch_calls["grad"] += 1
            else:
                dispatch_calls["no_grad"] += 1
            return original_dispatch(
                self, metadata, *expert_weights, runtime_weight_dtype=runtime_weight_dtype
            )

        monkeypatch.setattr(dispatcher_cls, "dispatch", counted_dispatch)

        dtype = torch.bfloat16 if expert_weight_backend == "hybridep" else torch.float32
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(_make_config(dtype, 1, balanced=False), _submodules(), layer_number=1)
        balanced_config = _make_config(
            dtype,
            1,
            balanced=True,
            recompute_expert_dispatch=True,
            expert_weight_backend=expert_weight_backend,
            grad_combine_dtype=(
                "param_dtype" if expert_weight_backend == "hybridep" else "fp32"
            ),
        )
        balanced = balanced_cls(balanced_config, _submodules(), layer_number=1)
        assert isinstance(balanced.expert_weight_dispatcher, dispatcher_cls)
        baseline.cuda()
        balanced.cuda()
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)

        baseline_output = _run_layer(baseline, hidden, output_grad)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        if expert_weight_backend == "hybridep":
            _assert_home_expert_grads_close(baseline, balanced, rtol=2e-2, atol=2e-2)
        else:
            _assert_home_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)
        assert dispatch_calls["no_grad"] == 2
        assert dispatch_calls["grad"] == 2
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,error_match",
    [
        ({"expert_model_parallel_size": 1}, "expert_model_parallel_size > 1"),
        ({"moe_num_spare_experts": None}, "positive moe_num_spare_experts"),
        (
            {"moe_balance_expert_weight_dispatch_backend": "invalid"},
            "expert_weight_dispatch_backend",
        ),
        (
            {"moe_balance_expert_weight_grad_combine_dtype": "invalid"},
            "expert_weight_grad_combine_dtype",
        ),
        (
            {"moe_balance_expert_weight_dispatch_backend": "hybridep"},
            "grad_combine_dtype='param_dtype'",
        ),
        (
            {
                "moe_balance_expert_weight_dispatch_backend": "hybridep",
                "moe_balance_expert_weight_grad_combine_dtype": "param_dtype",
                "moe_grouped_gemm": True,
            },
            "TEGroupedMLP runtime main_grad foldback",
        ),
        (
            {"moe_grouped_gemm": True, "moe_balance_recompute_expert_dispatch": True},
            "recompute_expert_dispatch",
        ),
        ({"moe_num_spare_experts": 8}, "one spare per EP rank"),
        ({"moe_token_dispatcher_type": "allgather"}, "moe_token_dispatcher_type='alltoall'"),
        ({"cuda_graph_impl": "local"}, "CUDA Graph"),
        ({"fp8": "hybrid"}, "FP8 or FP4"),
        ({"moe_router_padding_for_quantization": True}, "quantization router padding"),
        ({"moe_paged_stash": True}, "moe_paged_stash"),
        ({"moe_shared_expert_overlap": True}, "shared expert overlap"),
        ({"overlap_moe_expert_parallel_comm": True}, "MoE EP communication overlap"),
    ],
)
def test_balanced_moe_layer_config_guards(overrides, error_match):
    _require_distributed_cuda()
    with pytest.raises(ValueError, match=error_match):
        TransformerConfig(**_balanced_config_kwargs(**overrides))


@pytest.mark.internal
def test_module_spec_selects_balanced_layer():
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        _, balanced_cls = _require_balanced_moe_layer()
        dispatcher_module = importlib.import_module(
            "megatron.core.transformer.moe.expert_weight_dispatcher"
        )
        builder = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp

        regular_layer = builder(config=_make_config(torch.float32, 1, balanced=False))
        balanced_layer = builder(config=_make_config(torch.float32, 1, balanced=True))

        assert isinstance(regular_layer, MoELayer)
        assert not isinstance(regular_layer, balanced_cls)
        assert isinstance(balanced_layer, balanced_cls)
        assert "Balanced" not in type(balanced_layer.router).__name__
        assert type(balanced_layer.token_dispatcher).__name__ == "MoEAlltoAllTokenDispatcher"
        assert isinstance(
            balanced_layer.expert_weight_dispatcher,
            dispatcher_module.AllToAllExpertWeightDispatcher,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_balanced_moe_layer_symmetric_memory_backend_selection():
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        _, balanced_cls = _require_balanced_moe_layer()
        symm_dispatcher_cls = _require_symmetric_memory_backend()

        balanced = balanced_cls(
            _make_config(torch.float32, 1, balanced=True, expert_weight_backend="symmetric_memory"),
            _submodules(),
            layer_number=1,
        )

        assert isinstance(balanced.expert_weight_dispatcher, symm_dispatcher_cls)
        assert type(balanced.token_dispatcher).__name__ == "MoEAlltoAllTokenDispatcher"
        assert "symmetric" not in " ".join(balanced.state_dict().keys()).lower()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_balanced_moe_layer_symmetric_memory_backend_fails_fast_when_unavailable(monkeypatch):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        _, balanced_cls = _require_balanced_moe_layer()
        dispatcher_module = importlib.import_module(
            "megatron.core.transformer.moe.expert_weight_dispatcher"
        )
        monkeypatch.setattr(
            dispatcher_module.SymmetricMemoryExpertWeightDispatcher,
            "availability_error",
            staticmethod(lambda: "simulated missing symmetric memory support"),
        )

        with pytest.raises(RuntimeError, match="symmetric_memory expert-weight dispatch"):
            balanced_cls(
                _make_config(
                    torch.float32, 1, balanced=True, expert_weight_backend="symmetric_memory"
                ),
                _submodules(),
                layer_number=1,
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("mode", ["forced_move", "planner"])
def test_balanced_moe_layer_symmetric_memory_backend_parity(monkeypatch, mode):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        symm_dispatcher_cls = _require_symmetric_memory_backend()
        if mode != "planner":
            _patch_planner(monkeypatch, balanced_module, mode)

        dtype = torch.float32
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(_make_config(dtype, 1, balanced=False), _submodules(), layer_number=1)
        balanced = balanced_cls(
            _make_config(dtype, 1, balanced=True, expert_weight_backend="symmetric_memory"),
            _submodules(),
            layer_number=1,
        )
        assert isinstance(balanced.expert_weight_dispatcher, symm_dispatcher_cls)

        baseline.cuda()
        balanced.cuda()
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)

        baseline_output = _run_layer(baseline, hidden, output_grad)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        _assert_home_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)
        assert balanced.last_debug_stats is not None
        assert balanced.last_debug_stats.num_global_active_spare_slots > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    "feature,planner_mode", [("padding_mask", "no_move"), ("shared_expert", "forced_move")]
)
def test_balanced_moe_layer_padding_mask_and_shared_expert_parity(
    monkeypatch, feature, planner_mode
):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        _patch_planner(monkeypatch, balanced_module, planner_mode)

        dtype = torch.float32
        shared_expert = feature == "shared_expert"
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(
            _make_config(dtype, 1, balanced=False, shared_expert=shared_expert),
            _submodules(),
            layer_number=1,
        )
        balanced = balanced_cls(
            _make_config(dtype, 1, balanced=True, shared_expert=shared_expert),
            _submodules(),
            layer_number=1,
        )
        baseline.cuda()
        balanced.cuda()
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)
        forward_kwargs = {}
        if feature == "padding_mask":
            padding_mask = torch.ones(2, 6, device="cuda", dtype=torch.bool)
            padding_mask[:, -2:] = False
            forward_kwargs["padding_mask"] = padding_mask

        baseline_output = _run_layer(baseline, hidden, output_grad, **forward_kwargs)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad, **forward_kwargs)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        _assert_home_expert_grads_close(baseline, balanced)
        _assert_shared_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)

        if feature == "shared_expert":
            assert balanced.shared_experts is not None
            assert balanced.token_dispatcher.shared_experts is None
            assert balanced.last_debug_stats is not None
            assert balanced.last_debug_stats.num_global_active_spare_slots > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TE, reason="TEGroupedMLP parity requires Transformer Engine.")
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TEGroupedMLP is only supported in TE 1.9.0.dev0 and later.",
)
@pytest.mark.parametrize(
    "mode,topk,dtype",
    [
        ("no_move", 1, torch.bfloat16),
        ("forced_move", 1, torch.bfloat16),
        ("planner", 1, torch.bfloat16),
    ],
)
def test_balanced_moe_layer_te_grouped_parity(monkeypatch, mode, topk, dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        if mode != "planner":
            _patch_planner(monkeypatch, balanced_module, mode)

        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline_config = _make_config(dtype, topk, balanced=False, grouped=True)
        baseline = MoELayer(baseline_config, _submodules(grouped=True), layer_number=1)
        assert isinstance(baseline.experts, TEGroupedMLP)

        balanced_config = _make_config(dtype, topk, balanced=True, grouped=True)
        balanced = balanced_cls(balanced_config, _submodules(grouped=True), layer_number=1)
        assert isinstance(balanced.experts, TEGroupedMLP)

        baseline.cuda().to(dtype=dtype)
        balanced.cuda().to(dtype=dtype)
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)

        baseline_output = _run_layer(baseline, hidden, output_grad)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        _assert_home_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)

        if mode != "no_move":
            assert balanced.last_debug_stats is not None
            assert balanced.last_debug_stats.num_global_active_spare_slots > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(
    not HAVE_TE, reason="TEGroupedMLP runtime wgrad test requires Transformer Engine."
)
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TEGroupedMLP is only supported in TE 1.9.0.dev0 and later.",
)
@pytest.mark.parametrize("expert_weight_backend", ["all_to_all", "symmetric_memory"])
def test_balanced_moe_layer_te_grouped_runtime_weights_use_te_fp32_main_grad(
    monkeypatch, expert_weight_backend
):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        if expert_weight_backend == "symmetric_memory":
            _require_symmetric_memory_backend()
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        _patch_planner(monkeypatch, balanced_module, "forced_move")

        dtype = torch.bfloat16
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        balanced_config = _make_config(
            dtype, 1, balanced=True, grouped=True, expert_weight_backend=expert_weight_backend
        )
        balanced_config.gradient_accumulation_fusion = True
        balanced = balanced_cls(balanced_config, _submodules(grouped=True), layer_number=1)
        assert isinstance(balanced.experts, TEGroupedMLP)
        assert balanced_config.gradient_accumulation_fusion
        assert balanced.experts.linear_fc1.fuse_wgrad_accumulation
        assert balanced.experts.linear_fc2.fuse_wgrad_accumulation

        balanced.cuda().to(dtype=dtype)
        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        output_grad = torch.randn_like(hidden)
        _run_layer(balanced, hidden, output_grad)

        assert hidden.grad is not None
        assert balanced._last_runtime_spare_main_grad_dtypes
        assert all(
            grad_dtype is torch.float32
            for grad_dtype in balanced._last_runtime_spare_main_grad_dtypes
        )
        local_spare_main_grad_sum = torch.tensor(
            sum(balanced._last_runtime_spare_main_grad_abs_sums), device=hidden.device
        )
        torch.distributed.all_reduce(
            local_spare_main_grad_sum, op=torch.distributed.ReduceOp.SUM
        )
        assert local_spare_main_grad_sum.item() > 0.0

        local_home_main_grad_sum = torch.tensor(0.0, device=hidden.device)
        for param in balanced.experts.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is None:
                continue
            assert main_grad.dtype is torch.float32
            local_home_main_grad_sum += main_grad.abs().sum()
        torch.distributed.all_reduce(local_home_main_grad_sum, op=torch.distributed.ReduceOp.SUM)
        assert local_home_main_grad_sum.item() > 0.0
        _assert_no_spare_parameters(balanced)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TE, reason="TEGroupedMLP SymmMem parity requires Transformer Engine.")
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TEGroupedMLP is only supported in TE 1.9.0.dev0 and later.",
)
def test_balanced_moe_layer_te_grouped_symmetric_memory_parity(monkeypatch):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        _require_symmetric_memory_backend()
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        _patch_planner(monkeypatch, balanced_module, "forced_move")

        dtype = torch.bfloat16
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline_config = _make_config(dtype, 1, balanced=False, grouped=True)
        baseline = MoELayer(baseline_config, _submodules(grouped=True), layer_number=1)

        balanced_config = _make_config(
            dtype,
            1,
            balanced=True,
            grouped=True,
            expert_weight_backend="symmetric_memory",
        )
        balanced = balanced_cls(balanced_config, _submodules(grouped=True), layer_number=1)
        assert isinstance(baseline.experts, TEGroupedMLP)
        assert isinstance(balanced.experts, TEGroupedMLP)

        baseline.cuda().to(dtype=dtype)
        balanced.cuda().to(dtype=dtype)
        _copy_home_state(baseline, balanced)

        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        balanced_hidden = hidden.detach().clone().requires_grad_(True)
        output_grad = torch.randn_like(hidden)

        baseline_output = _run_layer(baseline, hidden, output_grad)
        balanced_output = _run_layer(balanced, balanced_hidden, output_grad)

        torch.testing.assert_close(balanced_output, baseline_output)
        torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
        _assert_router_grads_close(baseline, balanced)
        _assert_home_expert_grads_close(baseline, balanced)
        _assert_no_spare_parameters(balanced)
        assert balanced._last_runtime_spare_main_grad_dtypes
        assert balanced._last_runtime_folded_home_grad_dtypes
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(
    not HAVE_TE, reason="TEGroupedMLP runtime wgrad combine-dtype test requires Transformer Engine."
)
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TEGroupedMLP is only supported in TE 1.9.0.dev0 and later.",
)
def test_balanced_moe_layer_te_grouped_runtime_weight_grad_combine_dtype_is_configurable(
    monkeypatch,
):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        balanced_module, balanced_cls = _require_balanced_moe_layer()
        _patch_planner(monkeypatch, balanced_module, "forced_move")

        dtype = torch.bfloat16
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        balanced_config = _make_config(
            dtype,
            1,
            balanced=True,
            grouped=True,
            grad_combine_dtype="bf16",
        )
        balanced_config.gradient_accumulation_fusion = True
        balanced = balanced_cls(balanced_config, _submodules(grouped=True), layer_number=1)
        assert isinstance(balanced.experts, TEGroupedMLP)

        balanced.cuda().to(dtype=dtype)
        hidden = torch.randn(6, 2, 16, device="cuda", dtype=dtype, requires_grad=True)
        output_grad = torch.randn_like(hidden)
        _run_layer(balanced, hidden, output_grad)

        assert balanced._last_runtime_spare_main_grad_dtypes
        assert all(
            grad_dtype is torch.float32
            for grad_dtype in balanced._last_runtime_spare_main_grad_dtypes
        )
        assert balanced._last_runtime_folded_home_grad_dtypes
        assert all(
            grad_dtype is torch.bfloat16
            for grad_dtype in balanced._last_runtime_folded_home_grad_dtypes
        )
        for param in balanced.experts.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                assert main_grad.dtype is torch.float32
    finally:
        Utils.destroy_model_parallel()
