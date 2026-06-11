# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""General utilities."""
import json
import os
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch

from megatron.core._rank_utils import safe_get_rank as _safe_get_rank
from megatron.core._slurm_utils import resolve_slurm_local_rank
from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support
from megatron.core.msc_utils import MultiStorageClientFeature, open_file

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_data_parallel_group_if_dtensor,
    is_torch_min_version,
    to_local_if_dtensor,
    unwrap_model,
)
from megatron.training import get_adlr_autoresume, get_args, get_timers


def calc_params_l2_norm(model, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    args = get_args()
    if not isinstance(model, list):
        model = [model]

    if getattr(args, 'use_megatron_fsdp', False):
        # All Megatron FSDP parameters are expected to be PyTorch DTensor.
        # params_data is a dict of device_mesh -> list of local tensors.
        params = []
        for model_chunk in model:
            model_chunk.stop_communication()
            for name, param in model_chunk.named_parameters():
                if not hasattr(param, "_local_tensor"):
                    raise RuntimeError(
                        f"Megatron FSDP requires parameters are PyTorch DTensor. "
                        f"Parameter {name} is not a DTensor."
                    )
                params.append(param)

        return calc_dtensor_params_l2_norm(params)

    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, 'allreduce', True):
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                if args.bf16:
                    if not force_create_fp32_copy and hasattr(param, 'main_param'):
                        if getattr(param, 'main_param_sharded', False):
                            if param.main_param is not None:
                                sharded_params_data.append(param.main_param)
                        else:
                            moe_params_data.append(param.main_param)
                    else:
                        # Fallback to original logic of making a fp32 copy of the
                        # parameter if `.main_param` attribute is not available.
                        moe_params_data.append(param.data.float())
                else:
                    moe_params_data.append(param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if args.bf16:
                        if not force_create_fp32_copy and hasattr(param, 'main_param'):
                            if getattr(param, 'main_param_sharded', False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [params_data], False  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')

    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False,  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
    else:
        sharded_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    # Sum over all DP groups, including CP since distributed optimizer state is
    # sharded jointly over DP+CP.
    torch.distributed.all_reduce(
        sharded_norm_2,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
    )
    norm_2 += sharded_norm_2

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False,  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm

    # Account for MoE norm even if current rank doesn't have any expert params to prevent
    # hang in models with un-even numbers of MoE layers.
    # See details in https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/409
    else:
        moe_norm_2 = torch.zeros_like(norm_2)

    # Reduce norm across model parallel groups (dense and expert).
    # Dense params should sum across all model-parallel GPUs (tensor + pipeline).
    dense_reduce_group = mpu.get_model_parallel_group()
    ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
    # Expert params should sum across all model-parallel GPUs (expert + tensor + pipeline).
    expert_reduce_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
    ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    # If dense and expert reduce groups are the same, sum then reduce.
    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_norm_2
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
    # If dense and expert reduce groups are different, reduce then sum.
    else:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
        torch.distributed.all_reduce(
            moe_norm_2, op=torch.distributed.ReduceOp.SUM, group=expert_reduce_group
        )
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


def calc_dtensor_params_l2_norm(params):
    """Calculate l2 norm of DTensor parameters."""
    params_data = defaultdict(list)
    for param in params:
        params_data[param._spec].append(param._local_tensor)

    total_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device='cuda')
    for dtensor_spec, local_tensors in params_data.items():
        local_tensors = [t for t in local_tensors if t.numel() > 0]
        if len(local_tensors) == 0:
            norm = torch.zeros((1,), dtype=torch.float32, device='cuda')
        else:
            norm, _ = multi_tensor_applier(
                multi_tensor_l2norm,
                dummy_overflow_buf,
                [local_tensors],
                False,  # no per-parameter norm.
            )
        norm_2 = norm * norm
        for pg, placement in zip(
            dtensor_spec.device_mesh.get_all_groups(), dtensor_spec.placements
        ):
            if placement.is_shard():
                torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=pg)
            elif placement.is_replicate():
                # Replicated parameters are already summed across all ranks.
                pass
            else:
                raise RuntimeError(f"Unsupported placement {placement} for Megatron FSDP.")
        total_norm_2 += norm_2

    return total_norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / mpu.get_data_parallel_group().size()

    return averaged_losses


def reduce_max_stat_across_model_parallel_group(stat: float) -> float | None:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        # No rank has a valid stat, so return None to indicate that it is None across all ranks.
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input.item())


_NCCL_MEMORY_STATS_MIN_TORCH_VERSION = "2.12.0a0"


def _supports_nccl_memory_stats():
    """Return whether this PyTorch version may expose NCCL backend memory stats."""
    try:
        return is_torch_min_version(_NCCL_MEMORY_STATS_MIN_TORCH_VERSION)
    except Exception:
        return False


def _get_nccl_memory_stats():
    """Return NCCL communicator memory stats in bytes, if the runtime exposes them."""
    if (
        not _supports_nccl_memory_stats()
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or not torch.cuda.is_available()
    ):
        return {}

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        backend = torch.distributed.distributed_c10d._get_default_group()._get_backend(device)
        if getattr(backend, "_get_backend_name", lambda: None)() == "nccl":
            memory_stats = getattr(backend, "memory_stats", None)
            if callable(memory_stats):
                return memory_stats() or {}
    except Exception:
        pass
    return {}


def _get_nccl_memory_total_mb(mega_bytes):
    """Return NCCL communicator memory total in MB, if available."""
    total = _get_nccl_memory_stats().get("total")
    if total is None:
        return None
    return round(total / mega_bytes, 2)


def _memory_phase_logging_enabled():
    value = os.environ.get("MCORE_LOG_MEMORY_PHASES", "")
    return value.lower() in ("1", "true", "yes", "on")


def _memory_op_first_use_logging_enabled():
    value = os.environ.get("MCORE_LOG_MEMORY_OP_FIRST_USE", "")
    return value.lower() in ("1", "true", "yes", "on")


def _memory_op_first_use_sync_enabled():
    value = os.environ.get("MCORE_LOG_MEMORY_OP_FIRST_USE_SYNC", "1")
    return value.lower() not in ("0", "false", "no", "off")


def _memory_op_first_use_backward_enabled():
    value = os.environ.get("MCORE_LOG_MEMORY_OP_FIRST_USE_BACKWARD", "1")
    return value.lower() not in ("0", "false", "no", "off")


def _memory_op_first_use_scope():
    scope = os.environ.get("MCORE_LOG_MEMORY_OP_FIRST_USE_SCOPE", "category").lower()
    if scope not in ("category", "module"):
        return "category"
    return scope


_MEMORY_OP_FIRST_USE_STATE = {
    "installed": False,
    "seen": set(),
    "handles": [],
    "wrapped_methods": [],
    "last_residual_mb": None,
    "trace_start_residual_mb": None,
    "context": {},
}


def _memory_report_sample(mega_bytes, include_device_memory_used):
    """Return memory report fields in MB."""
    sample = {
        "allocated_mb": round(torch.cuda.memory_allocated() / mega_bytes, 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / mega_bytes, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / mega_bytes, 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / mega_bytes, 2),
        "device_memory_used_mb": None,
        "device_minus_reserved_mb": None,
        "nccl_memory_status": "unavailable",
        "nccl_memory_total_mb": None,
        "outside_reserved_residual_mb": None,
    }

    if include_device_memory_used:
        try:
            device_memory_used_mb = torch.cuda.device_memory_used() / mega_bytes
            sample["device_memory_used_mb"] = round(device_memory_used_mb, 2)
            sample["device_minus_reserved_mb"] = round(
                device_memory_used_mb - sample["reserved_mb"], 2
            )
        except Exception:
            pass

    nccl_memory_total_mb = _get_nccl_memory_total_mb(mega_bytes)
    if nccl_memory_total_mb is not None:
        sample["nccl_memory_status"] = "available"
        sample["nccl_memory_total_mb"] = nccl_memory_total_mb
        if sample["device_memory_used_mb"] is not None:
            sample["outside_reserved_residual_mb"] = round(
                sample["device_memory_used_mb"]
                - sample["reserved_mb"]
                - sample["nccl_memory_total_mb"],
                2,
            )

    return sample


def _format_memory_report_sample(sample, include_outside_reserved_residual=False):
    string = f" | allocated: {sample['allocated_mb']:.2f}"
    string += f" | max allocated: {sample['max_allocated_mb']:.2f}"
    string += f" | reserved: {sample['reserved_mb']:.2f}"
    string += f" | max reserved: {sample['max_reserved_mb']:.2f}"
    if sample["device_memory_used_mb"] is not None:
        string += f" | total device memory used: {sample['device_memory_used_mb']:.2f}"
    if sample["nccl_memory_total_mb"] is not None:
        string += f" | nccl memory (MB): total: {sample['nccl_memory_total_mb']:.2f}"
    if include_outside_reserved_residual and sample["outside_reserved_residual_mb"] is not None:
        string += f" | outside reserved residual: {sample['outside_reserved_residual_mb']:.2f}"
    return string


def report_memory(name):
    """Simple GPU memory report."""
    args = get_args()
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += _format_memory_report_sample(
        _memory_report_sample(mega_bytes, args.log_device_memory_used)
    )
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def report_memory_phase(label):
    """Report a gated structured memory sample for residual attribution."""
    if not _memory_phase_logging_enabled():
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if not torch.cuda.is_available():
        return

    mega_bytes = 1024.0 * 1024.0
    sample = _memory_report_sample(mega_bytes, include_device_memory_used=True)
    sample["label"] = label
    sample["rank"] = torch.distributed.get_rank()
    if mpu.get_data_parallel_rank() == 0:
        string = f"(memory phase: {label}) memory (MB)"
        string += _format_memory_report_sample(sample, include_outside_reserved_residual=True)
        print("[Rank {}] {}".format(sample["rank"], string), flush=True)
        print("MCORE_MEMORY_PHASE " + json.dumps(sample, sort_keys=True), flush=True)


def _should_report_memory_op_first_use():
    if not _memory_op_first_use_logging_enabled():
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return mpu.get_data_parallel_rank() == 0
    except Exception:
        return False


def _memory_op_first_use_key(stage, category, module_name):
    if _memory_op_first_use_scope() == "module":
        return (stage, module_name)
    return (stage, category)


def _memory_op_category(module_name, module_type):
    name = module_name.lower()
    module_type_lower = module_type.lower()

    if "self_attention.linear_qkv" in name:
        return "attention.linear_qkv"
    if "self_attention.core_attention" in name or "dotproductattention" in module_type_lower:
        return "attention.core_attention"
    if "self_attention.linear_proj" in name:
        return "attention.linear_proj"
    if "self_attention" in name and "q_layernorm" in name:
        return "attention.q_layernorm"
    if "self_attention" in name and "k_layernorm" in name:
        return "attention.k_layernorm"
    if name.endswith("input_layernorm"):
        return "layernorm.input"
    if name.endswith("pre_mlp_layernorm"):
        return "layernorm.pre_mlp"
    if "layernorm" in module_type_lower or module_type_lower == "tenorm":
        return "layernorm"

    if ".mlp.router" in name or "router" in module_type_lower:
        return "moe.router"
    if ".mlp.shared_experts" in name:
        return "moe.shared_experts"
    if ".mlp.experts" in name or "groupedmlp" in module_type_lower:
        return "moe.experts"
    if "sequentialmlp" in module_type_lower:
        return "moe.sequential_experts"
    if "moelayer" in module_type_lower:
        return "moe.layer"

    if ".mlp.linear_fc1" in name:
        return "mlp.linear_fc1"
    if ".mlp.linear_fc2" in name:
        return "mlp.linear_fc2"
    if module_type_lower == "mlp":
        return "mlp.block"

    if "selfattention" in module_type_lower:
        return "attention.block"
    if "columnparallellinear" in module_type_lower:
        return "linear.column_parallel"
    if "rowparallellinear" in module_type_lower:
        return "linear.row_parallel"

    return None


def _iter_first_tensor(value):
    if torch.is_tensor(value):
        if value.requires_grad:
            yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_first_tensor(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_first_tensor(item)


def _report_memory_op_first_use(stage, category, module_name, module_type):
    if not _should_report_memory_op_first_use():
        return

    state = _MEMORY_OP_FIRST_USE_STATE
    key = _memory_op_first_use_key(stage, category, module_name)
    if key in state["seen"]:
        return
    state["seen"].add(key)

    if _memory_op_first_use_sync_enabled():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    mega_bytes = 1024.0 * 1024.0
    sample = _memory_report_sample(mega_bytes, include_device_memory_used=True)
    sample["event"] = stage
    sample["category"] = category
    sample["module_name"] = module_name
    sample["module_type"] = module_type
    sample["rank"] = torch.distributed.get_rank()
    sample.update(state["context"])

    residual = sample["outside_reserved_residual_mb"]
    if residual is not None:
        if state["trace_start_residual_mb"] is None:
            state["trace_start_residual_mb"] = residual
        if state["last_residual_mb"] is None:
            sample["delta_from_previous_outside_reserved_residual_mb"] = 0.0
        else:
            sample["delta_from_previous_outside_reserved_residual_mb"] = round(
                residual - state["last_residual_mb"], 2
            )
        sample["delta_from_trace_start_outside_reserved_residual_mb"] = round(
            residual - state["trace_start_residual_mb"], 2
        )
        state["last_residual_mb"] = residual
    else:
        sample["delta_from_previous_outside_reserved_residual_mb"] = None
        sample["delta_from_trace_start_outside_reserved_residual_mb"] = None

    print("MCORE_MEMORY_OP_FIRST_USE " + json.dumps(sample, sort_keys=True), flush=True)


def _register_memory_op_tensor_hook(tensor, stage, category, module_name, module_type):
    key = _memory_op_first_use_key(stage, category, module_name)
    if key in _MEMORY_OP_FIRST_USE_STATE["seen"]:
        return

    def hook(grad):
        _report_memory_op_first_use(stage, category, module_name, module_type)
        return grad

    try:
        tensor.register_hook(hook)
    except Exception:
        pass


def _make_memory_op_forward_pre_hook(module_name, module_type, category):
    def hook(module, inputs):
        if not _should_report_memory_op_first_use():
            return
        _report_memory_op_first_use("forward_pre", category, module_name, module_type)
        if _memory_op_first_use_backward_enabled():
            for tensor in _iter_first_tensor(inputs):
                _register_memory_op_tensor_hook(
                    tensor, "backward_post", category, module_name, module_type
                )
                break

    return hook


def _make_memory_op_forward_hook(module_name, module_type, category):
    def hook(module, inputs, output):
        if not _should_report_memory_op_first_use():
            return
        _report_memory_op_first_use("forward_post", category, module_name, module_type)
        if _memory_op_first_use_backward_enabled():
            for tensor in _iter_first_tensor(output):
                _register_memory_op_tensor_hook(
                    tensor, "backward_pre", category, module_name, module_type
                )
                break

    return hook


def _wrap_memory_op_method(owner, method_name, category, qualified_name):
    wrapped_methods = getattr(owner, "_mcore_memory_op_first_use_wrapped_methods", set())
    if method_name in wrapped_methods:
        return False

    original = getattr(owner, method_name, None)
    if not callable(original):
        return False

    module_type = owner.__class__.__name__

    def wrapped(*args, **kwargs):
        _report_memory_op_first_use("call_pre", category, qualified_name, module_type)
        result = original(*args, **kwargs)
        _report_memory_op_first_use("call_post", category, qualified_name, module_type)
        return result

    setattr(owner, method_name, wrapped)
    wrapped_methods.add(method_name)
    setattr(owner, "_mcore_memory_op_first_use_wrapped_methods", wrapped_methods)
    _MEMORY_OP_FIRST_USE_STATE["wrapped_methods"].append(qualified_name)
    return True


def _install_memory_dispatcher_method_hooks(module_name, dispatcher):
    for method_name in (
        "dispatch_preprocess",
        "token_dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "token_combine",
        "combine_postprocess",
    ):
        _wrap_memory_op_method(
            dispatcher,
            method_name,
            f"moe.dispatcher.{method_name}",
            f"{module_name}.token_dispatcher.{method_name}",
        )


def set_memory_op_first_use_context(**context):
    """Set optional context fields attached to first-use memory trace events."""
    if not _memory_op_first_use_logging_enabled():
        return
    _MEMORY_OP_FIRST_USE_STATE["context"] = {
        key: value for key, value in context.items() if value is not None
    }


def install_memory_op_first_use_hooks(model):
    """Install gated first-use memory hooks on selected training ops/modules."""
    if not _should_report_memory_op_first_use():
        return

    state = _MEMORY_OP_FIRST_USE_STATE
    if state["installed"]:
        return
    state["installed"] = True

    model_chunks = model if isinstance(model, list) else [model]
    unwrapped_chunks = unwrap_model(model_chunks)
    if not isinstance(unwrapped_chunks, list):
        unwrapped_chunks = [unwrapped_chunks]

    hooked_modules = 0
    for chunk_idx, model_chunk in enumerate(unwrapped_chunks):
        for module_name, module in model_chunk.named_modules():
            qualified_name = f"model_chunk{chunk_idx}.{module_name}" if module_name else (
                f"model_chunk{chunk_idx}"
            )
            module_type = module.__class__.__name__
            category = _memory_op_category(module_name, module_type)
            if category is not None:
                state["handles"].append(
                    module.register_forward_pre_hook(
                        _make_memory_op_forward_pre_hook(qualified_name, module_type, category)
                    )
                )
                state["handles"].append(
                    module.register_forward_hook(
                        _make_memory_op_forward_hook(qualified_name, module_type, category)
                    )
                )
                hooked_modules += 1

            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is not None:
                _install_memory_dispatcher_method_hooks(qualified_name, dispatcher)

    if _should_report_memory_op_first_use():
        sample = {
            "rank": torch.distributed.get_rank(),
            "hooked_modules": hooked_modules,
            "wrapped_methods": len(state["wrapped_methods"]),
            "scope": _memory_op_first_use_scope(),
            "sync": _memory_op_first_use_sync_enabled(),
            "backward": _memory_op_first_use_backward_enabled(),
        }
        print("MCORE_MEMORY_OP_TRACE_INSTALLED " + json.dumps(sample, sort_keys=True), flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel)
            )
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    pad_token,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    pad_mask_loss,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    if pad_mask_loss:
        loss_mask[data == pad_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = (
                position_ids[b, data[b] == eod_token] & position_ids[b, data[b] == pad_token]
            )
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    else:
        if _safe_get_rank() == 0:
            print(message, flush=True)


def warn_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, warn only on rank 0."""
    if rank is not None:
        if rank == 0:
            warnings.warn(message)
    else:
        if _safe_get_rank() == 0:
            warnings.warn(message)


def is_rank0():
    """Returns true if called in the rank0, false otherwise."""
    return _safe_get_rank() == 0


def is_last_rank():
    """Returns true if called on last rank, false otherwise."""
    assert torch.distributed.is_initialized()
    return _safe_get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized() and torch.distributed.get_backend() != 'fake':
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_hybrid_model(args):
    """Returns True if the model is a hybrid Mamba-Transformer model."""
    return args.hybrid_layer_pattern is not None


def is_first_or_last_pipeline_stage(vp_stage):
    """Return True if on first or last pipeline stage, taking into account virtual
    pipeline parallelism."""
    ignore_virtual = True
    if vp_stage is not None:
        ignore_virtual = False
    return mpu.is_pipeline_first_stage(
        ignore_virtual=ignore_virtual, vp_stage=vp_stage
    ) or mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)


def get_device_arch_version():
    """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    return torch.cuda.get_device_properties(torch.device("cuda:0")).major


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = args.data_path is not None or args.data_args_path is not None
    use_per_split_data_path = (
        any(
            elt is not None
            for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]
        )
        or args.per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if args.data_args_path is not None:
            assert args.data_path is None
            with open_file(args.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert args.data_path is not None
            blend = get_blend_from_list(args.data_path)
    elif use_per_split_data_path:
        if args.per_split_data_args_path is not None:
            with open_file(args.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        assert data_iterator is not None
        data = next(data_iterator)
        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': (
                None
                if "attention_mask" not in data
                else data["attention_mask"].cuda(non_blocking=True)
            ),
            'position_ids': data["position_ids"].cuda(non_blocking=True),
            'cu_seqlens': (
                None if "cu_seqlens" not in data else data["cu_seqlens"].cuda(non_blocking=True)
            ),
            'max_seqlen': (
                None if "max_seqlen" not in data else data["max_seqlen"].cuda(non_blocking=True)
            ),
            'local_cp_size': (
                None
                if "local_cp_size" not in data
                else data["local_cp_size"].cuda(non_blocking=True)
            ),
        }

        def _broadcast_cu_seqlens(cu_seqlens):
            if getattr(args, 'cuda_graph_impl', 'none') == 'full_iteration':
                assert (
                    cu_seqlens is None
                ), "cu_seqlens is not supported with cuda_graph_impl=full_iteration"
                return
            dev = torch.cuda.current_device()
            n = 0 if cu_seqlens is None else int(cu_seqlens.numel())
            n_tensor = torch.empty(1, dtype=torch.int64, device=dev).fill_(n)
            _broadcast(n_tensor)

            if n == 0:
                buf = torch.empty(0, dtype=torch.int32, device=dev)
            else:
                assert isinstance(cu_seqlens, torch.Tensor)
                assert cu_seqlens.dtype == torch.int32
                assert cu_seqlens.shape[0] == 1, "micro-batch-size must be 1 for packing"
                buf = cu_seqlens.to(device=dev, non_blocking=True).contiguous()
            _broadcast(buf)

        if args.dynamic_context_parallel:
            seq_len = torch.tensor(
                batch['tokens'].shape[0], dtype=torch.int32, device=torch.cuda.current_device()
            )
            _broadcast(seq_len)

        if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast_cu_seqlens(batch['cu_seqlens'])
            _broadcast(batch['max_seqlen'])
            _broadcast(batch['local_cp_size'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast_cu_seqlens(batch['cu_seqlens'])
            _broadcast(batch['max_seqlen'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:
        if args.dynamic_context_parallel:
            seq_len = torch.tensor(0, dtype=torch.int32, device=torch.cuda.current_device())
            _broadcast(seq_len)
            shape = seq_len.item()
        else:
            shape = (args.micro_batch_size, args.seq_length)

        tokens = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty(shape, dtype=torch.float32, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            shape_attention_mask = (
                (args.micro_batch_size, 1, args.seq_length, args.seq_length)
                if not args.dynamic_context_parallel
                else (1, 1, shape[0], shape[0])
            )
            attention_mask = torch.empty(
                shape_attention_mask, dtype=torch.bool, device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        cu_seqlens = None
        if args.dynamic_context_parallel or args.sft:
            max_seqlen = torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
        else:
            max_seqlen = None

        local_cp_size = (
            torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
            if args.dynamic_context_parallel
            else None
        )

        def _broadcast_cu_seqlens():
            if getattr(args, 'cuda_graph_impl', 'none') == 'full_iteration':
                return None
            dev = torch.cuda.current_device()

            n = torch.empty((), dtype=torch.int64, device=dev)
            _broadcast(n)
            n = int(n.item())

            if n == 0:
                cu_seqlens = torch.empty(0, dtype=torch.int32, device=dev)
            else:
                cu_seqlens = torch.empty((args.micro_batch_size, n), dtype=torch.int32, device=dev)
            _broadcast(cu_seqlens)

            return cu_seqlens if n > 0 else None

        if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            cu_seqlens = _broadcast_cu_seqlens()
            _broadcast(max_seqlen)
            _broadcast(local_cp_size)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            cu_seqlens = _broadcast_cu_seqlens()
            _broadcast(max_seqlen)

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            tokens = None
            position_ids = None
            cu_seqlens = None
            max_seqlen = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
            'local_cp_size': local_cp_size,
        }

    return batch


def update_use_dist_ckpt(args):
    args.use_dist_ckpt = args.ckpt_format != "torch"


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Officially, torch suggests to_empty() for meta device materialization. Under the hood,
    torch.empty_like() is applied to all parameters or buffers (see _apply). This may
    accidently overwrite buffers with precomputed values during construction. Given the
    goal is to only materialize those tensors on meta device, this function checks the
    device first and only move the tensor to the destination if it is not on meta device.

    Args:
        module: The target module to apply this transformation.
        device: The desired device of the parameters
            and buffers in this module.
        recurse: Whether parameters and buffers of submodules should
            be recursively moved to the specified device.
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        else:
            return tensor.to(device)

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)


def get_nvtx_range():
    """Create an NVTX range context manager.

    Returns a context manager that:
    - Creates an NVTX range for profiling (nsight-systems compatible)
    - Optionally tracks time via Megatron timers when time=True

    Args (for returned context manager):
        msg: Name of the range/timer
        time: If True, also track with Megatron timers (default: False)
        log_level: Timer log level (0=always, 1=default, 2=verbose). Default: 1
    """
    from megatron.core.utils import nvtx_range_pop, nvtx_range_push

    @contextmanager
    def nvtx_range(msg, time=False, log_level=1):
        if time:
            timers = get_timers()
            timers(msg, log_level=log_level).start()
        try:
            nvtx_range_push(msg)
            yield
        finally:
            nvtx_range_pop(msg)
            if time:
                timers(msg, log_level=log_level).stop()

    return nvtx_range


def has_nvrx_installed():
    """Checks if nvidia-resiliency-ext is installed."""
    try:
        import nvidia_resiliency_ext

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def has_nvrx_checkpointing_async_support():
    """Checks whether the installed NVRx package exposes the async checkpointing API Megatron uses."""
    return has_nvrx_async_support()


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Fallback order:
    1. LOCAL_RANK environment variable (torchrun/torchelastic)
    2. SLURM_LOCALID environment variable (SLURM)
    3. Default: 0 (with warning)

    Returns:
        The local rank of the current process.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    slurm_local_rank = resolve_slurm_local_rank()
    if slurm_local_rank is not None:
        return slurm_local_rank

    warnings.warn(
        "Could not determine local rank from LOCAL_RANK or SLURM_LOCALID. Defaulting to local rank 0."
    )
    return 0
