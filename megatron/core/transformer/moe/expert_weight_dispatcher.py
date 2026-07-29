# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert-weight dispatch for BalancedMoELayer.

The dispatcher moves checkpoint-owned home expert weights into runtime spare
slots. Forward sends only active home-to-spare pairs across the EP group.
Backward explicitly folds spare gradients back to the owning home expert
parameters through the reverse communication pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from deep_ep import HybridEPBuffer
except ImportError:
    HybridEPBuffer = None

_HYBRIDEP_WEIGHT_TOKEN_ALIGNMENT = 64
_SYMMETRIC_MEMORY_BUFFER_SLOTS = 2
_EXPERT_WEIGHT_GRAD_COMBINE_DTYPE_POLICIES = ("fp32", "param_dtype", "bf16", "fp16")


def _expert_weight_grad_combine_dtype(
    weight_dtype: torch.dtype, policy: str = "fp32"
) -> torch.dtype:
    """Resolve the dtype used when folding spare weight gradients."""

    if policy == "fp32":
        if weight_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return weight_dtype
    if policy == "param_dtype":
        return weight_dtype
    if policy == "bf16":
        return torch.bfloat16
    if policy == "fp16":
        return torch.float16
    raise ValueError(
        "moe_balance_expert_weight_grad_combine_dtype must be one of "
        f"{_EXPERT_WEIGHT_GRAD_COMBINE_DTYPE_POLICIES}; got {policy!r}."
    )


def _expert_weight_grad_combine_policy(config: TransformerConfig) -> str:
    policy = getattr(config, "moe_balance_expert_weight_grad_combine_dtype", "fp32")
    if policy not in _EXPERT_WEIGHT_GRAD_COMBINE_DTYPE_POLICIES:
        raise ValueError(
            "moe_balance_expert_weight_grad_combine_dtype must be one of "
            f"{_EXPERT_WEIGHT_GRAD_COMBINE_DTYPE_POLICIES}; got {policy!r}."
        )
    return policy


def _runtime_weight_grad_edge_dtype(weight_dtype: torch.dtype, policy: str) -> torch.dtype:
    """Dtype for non-TE runtime-weight autograd edges."""

    return _expert_weight_grad_combine_dtype(weight_dtype, policy)


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
    send_local_home_indices: list[int]
    recv_local_spare_indices: list[int]
    active_local_home_indices: list[int]
    local_spare_sources: list[Optional[tuple[int, int]]]
    remote_spare_aliases: list[Optional[int]]
    unique_remote_source_indices: list[int]
    backward_grad_sources: list[tuple[int, int, int, int]]


@dataclass
class _SymmetricExpertWeightDispatchWorkspace:
    """Symmetric-memory buffers cached for one weight shape/dtype/device."""

    home_weight_buffer: torch.Tensor
    spare_grad_buffer: torch.Tensor
    home_weight_handle: Any
    spare_grad_handle: Any
    next_home_weight_slot: int = 0
    next_spare_grad_slot: int = 0


class _AllToAllExpertWeightDispatch(torch.autograd.Function):
    """Autograd bridge for home-weight to spare-slot dispatch."""

    @staticmethod
    def forward(
        ctx,
        ep_group: torch.distributed.ProcessGroup,
        ep_rank: int,
        ep_size: int,
        num_local_home_experts: int,
        num_local_spare_experts: int,
        input_splits: list[int],
        output_splits: list[int],
        send_local_home_indices: list[int],
        recv_local_spare_indices: list[int],
        runtime_weight_dtype: Optional[torch.dtype],
        grad_combine_dtype: torch.dtype,
        *local_home_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Send active home expert weights to local spare expert slots."""

        if len(local_home_weights) != num_local_home_experts:
            raise ValueError(
                f"Expected {num_local_home_experts} local home weights, "
                f"got {len(local_home_weights)}."
            )
        if not local_home_weights:
            raise ValueError("At least one local home weight is required.")

        reference_weight = local_home_weights[0]
        if runtime_weight_dtype is None:
            runtime_weight_dtype = reference_weight.dtype
        for weight in local_home_weights:
            if weight.shape != reference_weight.shape:
                raise ValueError(
                    "All expert weights in one dispatch call must have the same shape; "
                    f"got {tuple(weight.shape)} and {tuple(reference_weight.shape)}."
                )

        send_count = sum(input_splits)
        recv_count = sum(output_splits)
        if len(send_local_home_indices) != send_count:
            raise ValueError(
                "send_local_home_indices length must match the sum of input_splits; "
                f"got {len(send_local_home_indices)} and {send_count}."
            )
        if len(recv_local_spare_indices) != recv_count:
            raise ValueError(
                "recv_local_spare_indices length must match the sum of output_splits; "
                f"got {len(recv_local_spare_indices)} and {recv_count}."
            )

        if send_count:
            send_tensor = torch.stack(
                [local_home_weights[local_home_idx] for local_home_idx in send_local_home_indices],
                dim=0,
            ).contiguous()
        else:
            send_tensor = reference_weight.new_empty((0, *reference_weight.shape))
        recv_tensor = reference_weight.new_empty((recv_count, *reference_weight.shape))

        torch.distributed.all_to_all_single(
            recv_tensor,
            send_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
        )

        outputs: list[torch.Tensor] = [
            torch.zeros(
                reference_weight.shape,
                dtype=runtime_weight_dtype,
                device=reference_weight.device,
            )
            for _ in range(num_local_spare_experts)
        ]
        for recv_offset, local_spare_idx in enumerate(recv_local_spare_indices):
            outputs[local_spare_idx] = recv_tensor[recv_offset].clone().to(
                dtype=runtime_weight_dtype
            )

        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size
        ctx.num_local_home_experts = num_local_home_experts
        ctx.weight_shape = tuple(reference_weight.shape)
        ctx.weight_dtype = reference_weight.dtype
        ctx.grad_combine_dtype = grad_combine_dtype
        ctx.weight_device = reference_weight.device
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.send_local_home_indices = send_local_home_indices
        ctx.recv_local_spare_indices = recv_local_spare_indices

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        """Fold spare expert gradients back to their owning home experts."""

        grad_combine_dtype = ctx.grad_combine_dtype
        grad_send_tensors: list[torch.Tensor] = []
        for local_spare_idx in ctx.recv_local_spare_indices:
            grad_output = grad_outputs[local_spare_idx]
            if grad_output is None:
                grad_send_tensors.append(
                    torch.zeros(ctx.weight_shape, dtype=grad_combine_dtype, device=ctx.weight_device)
                )
            else:
                grad_send_tensors.append(grad_output.to(dtype=grad_combine_dtype))

        if grad_send_tensors:
            grad_send_tensor = torch.stack(grad_send_tensors, dim=0).contiguous()
        else:
            grad_send_tensor = torch.empty(
                (0, *ctx.weight_shape), dtype=grad_combine_dtype, device=ctx.weight_device
            )
        grad_recv_tensor = torch.empty(
            (sum(ctx.input_splits), *ctx.weight_shape),
            dtype=grad_combine_dtype,
            device=ctx.weight_device,
        )

        torch.distributed.all_to_all_single(
            grad_recv_tensor,
            grad_send_tensor,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.ep_group,
        )

        local_home_grads = [
            torch.zeros(ctx.weight_shape, dtype=grad_combine_dtype, device=ctx.weight_device)
            for _ in range(ctx.num_local_home_experts)
        ]
        for local_home_idx, grad in zip(ctx.send_local_home_indices, grad_recv_tensor):
            local_home_grads[local_home_idx].add_(grad)

        return (None, None, None, None, None, None, None, None, None, None, None, *local_home_grads)


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
        self.grad_combine_dtype_policy = _expert_weight_grad_combine_policy(config)

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

    def _grad_combine_dtype(self, weight_dtype: torch.dtype) -> torch.dtype:
        return _expert_weight_grad_combine_dtype(weight_dtype, self.grad_combine_dtype_policy)

    def _runtime_weight_grad_edge_dtype(self, weight_dtype: torch.dtype) -> torch.dtype:
        return _runtime_weight_grad_edge_dtype(weight_dtype, self.grad_combine_dtype_policy)

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

        send_local_home_indices_by_rank: list[list[int]] = [[] for _ in range(self.ep_size)]
        recv_local_spare_indices_by_rank: list[list[int]] = [[] for _ in range(self.ep_size)]
        active_local_home_indices: set[int] = set()
        local_spare_sources: list[Optional[tuple[int, int]]] = [
            None for _ in range(self.num_local_spare_experts)
        ]
        backward_grad_sources: list[tuple[int, int, int, int]] = []
        for spare_idx in range(self.num_spare_experts):
            home_indices = torch.where(expert_offloading_map[:, spare_idx])[0]
            if home_indices.numel() == 0:
                continue
            home_idx = int(home_indices.item())
            source_rank = home_idx // self.num_local_home_experts
            local_home_idx = home_idx % self.num_local_home_experts
            dest_rank = spare_idx // self.num_local_spare_experts
            local_spare_idx = spare_idx % self.num_local_spare_experts

            if source_rank == self.ep_rank:
                send_local_home_indices_by_rank[dest_rank].append(local_home_idx)
                backward_grad_sources.append(
                    (dest_rank, local_spare_idx, local_home_idx, local_spare_idx)
                )
                if dest_rank != self.ep_rank:
                    active_local_home_indices.add(local_home_idx)
            if dest_rank == self.ep_rank:
                recv_local_spare_indices_by_rank[source_rank].append(local_spare_idx)
                local_spare_sources[local_spare_idx] = (source_rank, local_home_idx)

        remote_spare_aliases: list[Optional[int]] = [
            None for _ in range(self.num_local_spare_experts)
        ]
        unique_remote_source_indices: list[int] = []
        representative_by_source: dict[tuple[int, int], int] = {}
        for local_spare_idx, source in enumerate(local_spare_sources):
            if source is None:
                continue
            source_rank, _ = source
            if source_rank == self.ep_rank:
                continue
            representative_local_spare_idx = representative_by_source.setdefault(
                source, local_spare_idx
            )
            remote_spare_aliases[local_spare_idx] = representative_local_spare_idx
            if representative_local_spare_idx == local_spare_idx:
                unique_remote_source_indices.append(local_spare_idx)

        return ExpertWeightDispatchMetadata(
            global_routing_map=expert_offloading_map,
            local_to_global_routing_map=local_to_global,
            global_to_local_routing_map=global_to_local,
            input_splits=local_to_global.sum(dim=(0, 2)).tolist(),
            output_splits=global_to_local.sum(dim=(1, 2)).tolist(),
            num_out_experts=int(local_to_global.sum().item()),
            has_experts_per_slot=global_to_local.sum(dim=(0, 1)),
            local_spare_home_indices=local_spare_home_indices,
            send_local_home_indices=[
                local_home_idx
                for per_rank_indices in send_local_home_indices_by_rank
                for local_home_idx in per_rank_indices
            ],
            recv_local_spare_indices=[
                local_spare_idx
                for per_rank_indices in recv_local_spare_indices_by_rank
                for local_spare_idx in per_rank_indices
            ],
            active_local_home_indices=sorted(active_local_home_indices),
            local_spare_sources=local_spare_sources,
            remote_spare_aliases=remote_spare_aliases,
            unique_remote_source_indices=unique_remote_source_indices,
            backward_grad_sources=backward_grad_sources,
        )

    def dispatch(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *expert_weights: torch.Tensor,
        runtime_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        """Return weights for this rank's local spare slots."""

        if not expert_weights:
            raise ValueError("At least one local home weight is required.")
        outputs = _AllToAllExpertWeightDispatch.apply(
            self.ep_group,
            self.ep_rank,
            self.ep_size,
            self.num_local_home_experts,
            self.num_local_spare_experts,
            metadata.input_splits,
            metadata.output_splits,
            metadata.send_local_home_indices,
            metadata.recv_local_spare_indices,
            runtime_weight_dtype,
            self._grad_combine_dtype(expert_weights[0].dtype),
            *expert_weights,
        )
        return list(outputs)

    def expert_dispatch(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *expert_weights: torch.Tensor,
        runtime_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        """Compatibility alias for the old Echo dispatcher method name."""

        return self.dispatch(
            metadata, *expert_weights, runtime_weight_dtype=runtime_weight_dtype
        )

    def fold_spare_gradients(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *local_spare_grads: torch.Tensor,
        reference_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        """Fold local spare-slot gradients back to local home expert gradients."""

        if not local_spare_grads:
            raise ValueError("At least one local spare gradient is required.")
        reference_grad = local_spare_grads[0]
        grad_combine_dtype = self._grad_combine_dtype(
            reference_weight_dtype or reference_grad.dtype
        )
        grad_send_tensors: list[torch.Tensor] = []
        for local_spare_idx in metadata.recv_local_spare_indices:
            grad_send_tensors.append(
                local_spare_grads[local_spare_idx].to(dtype=grad_combine_dtype)
            )

        if grad_send_tensors:
            grad_send_tensor = torch.stack(grad_send_tensors, dim=0).contiguous()
        else:
            grad_send_tensor = torch.empty(
                (0, *reference_grad.shape),
                dtype=grad_combine_dtype,
                device=reference_grad.device,
            )
        grad_recv_tensor = torch.empty(
            (sum(metadata.input_splits), *reference_grad.shape),
            dtype=grad_combine_dtype,
            device=reference_grad.device,
        )

        torch.distributed.all_to_all_single(
            grad_recv_tensor,
            grad_send_tensor,
            output_split_sizes=metadata.input_splits,
            input_split_sizes=metadata.output_splits,
            group=self.ep_group,
        )

        local_home_grads = [
            torch.zeros(reference_grad.shape, dtype=grad_combine_dtype, device=reference_grad.device)
            for _ in range(self.num_local_home_experts)
        ]
        for local_home_idx, grad in zip(metadata.send_local_home_indices, grad_recv_tensor):
            local_home_grads[local_home_idx].add_(grad)
        return local_home_grads


def _get_symmetric_memory_module():
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as exc:
        raise RuntimeError("torch.distributed._symmetric_memory is not importable.") from exc

    missing = [name for name in ("empty", "rendezvous") if not hasattr(symm_mem, name)]
    if missing:
        raise RuntimeError(
            "torch.distributed._symmetric_memory is missing required API(s): " + ", ".join(missing)
        )
    return symm_mem


def _enable_symmetric_memory_for_group(symm_mem, group: torch.distributed.ProcessGroup) -> None:
    group_name = getattr(group, "group_name", None)
    if group_name is not None and hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(group_name)


def _rendezvous_symmetric_buffer(
    symm_mem, tensor: torch.Tensor, group: torch.distributed.ProcessGroup
):
    try:
        return symm_mem.rendezvous(tensor, group)
    except TypeError:
        group_name = getattr(group, "group_name", group)
        return symm_mem.rendezvous(tensor, group=group_name)


class _SymmetricMemoryExpertWeightDispatch(torch.autograd.Function):
    """Autograd bridge for low-level SymmMem expert-weight dispatch."""

    @staticmethod
    def forward(
        ctx,
        dispatcher: "SymmetricMemoryExpertWeightDispatcher",
        metadata: ExpertWeightDispatchMetadata,
        runtime_weight_dtype: Optional[torch.dtype],
        grad_combine_dtype: torch.dtype,
        *local_home_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Read remote home expert weights through symmetric-memory handles."""

        dispatcher._validate_weights(local_home_weights)
        dispatcher._reset_debug_counters()
        reference_weight = local_home_weights[0]
        if runtime_weight_dtype is None:
            runtime_weight_dtype = reference_weight.dtype
        weight_numel = reference_weight.numel()
        workspace = dispatcher._get_workspace(
            reference_weight, grad_combine_dtype=grad_combine_dtype
        )
        home_weight_slot = dispatcher._next_home_weight_slot(workspace)
        home_weight_view = workspace.home_weight_buffer.view(
            _SYMMETRIC_MEMORY_BUFFER_SLOTS,
            dispatcher.num_local_home_experts,
            *reference_weight.shape,
        )[home_weight_slot]

        with torch.no_grad():
            for local_home_idx in metadata.active_local_home_indices:
                home_weight_view[local_home_idx].copy_(
                    local_home_weights[local_home_idx].detach()
                )
        dispatcher._debug_staged_home_experts = len(metadata.active_local_home_indices)

        workspace.home_weight_handle.barrier()
        dispatcher._debug_home_weight_barriers += 1

        outputs: list[torch.Tensor] = [
            torch.zeros(
                reference_weight.shape,
                dtype=runtime_weight_dtype,
                device=reference_weight.device,
            )
            for _ in range(dispatcher.num_local_spare_experts)
        ]
        representative_outputs: dict[int, torch.Tensor] = {}
        for local_spare_idx, source in enumerate(metadata.local_spare_sources):
            if source is None:
                continue

            source_rank, source_local_home = source
            if source_rank == dispatcher.ep_rank:
                output = local_home_weights[source_local_home].to(dtype=runtime_weight_dtype)
                outputs[local_spare_idx] = output
                continue

            representative_local_spare_idx = metadata.remote_spare_aliases[local_spare_idx]
            if representative_local_spare_idx is None:
                raise RuntimeError(
                    "Symmetric Memory expert-weight dispatch missing a representative slot for "
                    f"remote local spare {local_spare_idx}."
                )
            if representative_local_spare_idx not in representative_outputs:
                remote_weight = torch.empty_like(reference_weight)
                offset = (
                    home_weight_slot * dispatcher.num_local_home_experts + source_local_home
                ) * weight_numel
                dispatcher._copy_from_symmetric_peer(
                    remote_weight.contiguous().view(-1),
                    workspace.home_weight_handle,
                    peer=source_rank,
                    offset=offset,
                )
                output = remote_weight.to(dtype=runtime_weight_dtype)
                representative_outputs[representative_local_spare_idx] = output
            else:
                output = representative_outputs[representative_local_spare_idx].clone()
                dispatcher._debug_duplicate_remote_reads_avoided += 1
            outputs[local_spare_idx] = output
        dispatcher._debug_unique_remote_sources = len(representative_outputs)

        ctx.dispatcher = dispatcher
        ctx.dispatch_metadata = metadata
        ctx.workspace = workspace
        ctx.weight_shape = tuple(reference_weight.shape)
        ctx.weight_dtype = reference_weight.dtype
        ctx.grad_combine_dtype = grad_combine_dtype
        ctx.weight_device = reference_weight.device
        ctx.weight_numel = weight_numel

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        """Read remote spare gradients and accumulate local home gradients."""

        dispatcher: SymmetricMemoryExpertWeightDispatcher = ctx.dispatcher
        metadata: ExpertWeightDispatchMetadata = ctx.dispatch_metadata
        workspace: _SymmetricExpertWeightDispatchWorkspace = ctx.workspace
        grad_combine_dtype = ctx.grad_combine_dtype

        spare_grad_slot = dispatcher._next_spare_grad_slot(workspace)
        spare_grad_view = workspace.spare_grad_buffer.view(
            _SYMMETRIC_MEMORY_BUFFER_SLOTS,
            dispatcher.num_local_spare_experts,
            *ctx.weight_shape,
        )[spare_grad_slot]
        with torch.no_grad():
            spare_grad_view.zero_()
            for local_spare_idx in metadata.recv_local_spare_indices:
                grad_output = grad_outputs[local_spare_idx]
                if grad_output is not None:
                    spare_grad_view[local_spare_idx].copy_(
                        grad_output.detach().to(dtype=grad_combine_dtype)
                    )

        workspace.spare_grad_handle.barrier()
        dispatcher._debug_spare_grad_barriers += 1

        local_home_grads = [
            torch.zeros(ctx.weight_shape, dtype=grad_combine_dtype, device=ctx.weight_device)
            for _ in range(dispatcher.num_local_home_experts)
        ]

        dispatcher._debug_backward_schedule_entries = len(metadata.backward_grad_sources)
        for (
            dest_rank,
            dest_local_spare,
            source_local_home,
            dest_spare_offset,
        ) in metadata.backward_grad_sources:
            if source_local_home < 0 or source_local_home >= dispatcher.num_local_home_experts:
                raise RuntimeError(
                    "Symmetric Memory backward schedule has an invalid local home index: "
                    f"{source_local_home}."
                )
            if dest_rank < 0 or dest_rank >= dispatcher.ep_size:
                raise RuntimeError(
                    "Symmetric Memory backward schedule has an invalid destination rank: "
                    f"{dest_rank}."
                )
            if dest_local_spare < 0 or dest_local_spare >= dispatcher.num_local_spare_experts:
                raise RuntimeError(
                    "Symmetric Memory backward schedule has an invalid local spare index: "
                    f"{dest_local_spare}."
                )
            if dest_spare_offset < 0 or dest_spare_offset >= dispatcher.num_local_spare_experts:
                raise RuntimeError(
                    "Symmetric Memory backward schedule has an invalid spare-gradient offset: "
                    f"{dest_spare_offset}."
                )
            if dest_rank == dispatcher.ep_rank:
                local_home_grads[source_local_home].add_(spare_grad_view[dest_local_spare])
            else:
                tmp_grad = torch.empty(
                    ctx.weight_shape, dtype=grad_combine_dtype, device=ctx.weight_device
                )
                offset = (
                    spare_grad_slot * dispatcher.num_local_spare_experts + dest_spare_offset
                ) * ctx.weight_numel
                dispatcher._copy_from_symmetric_peer(
                    tmp_grad.contiguous().view(-1),
                    workspace.spare_grad_handle,
                    peer=dest_rank,
                    offset=offset,
                )
                local_home_grads[source_local_home].add_(tmp_grad)

        return (None, None, None, None, *local_home_grads)


class SymmetricMemoryExpertWeightDispatcher(AllToAllExpertWeightDispatcher):
    """Experimental low-level SymmMem expert-weight dispatcher.

    This backend intentionally uses low-level one-sided remote reads with
    symmetric staging buffers. It does not call ``all_to_all_single`` or SymmMem
    all-to-all helper ops.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ep_group: torch.distributed.ProcessGroup,
        num_home_experts: int,
        num_spare_experts: int,
    ) -> None:
        super().__init__(config, ep_group, num_home_experts, num_spare_experts)
        self._symm_mem = _get_symmetric_memory_module()
        _enable_symmetric_memory_for_group(self._symm_mem, ep_group)
        self._debug_low_level_get_calls = 0
        self._debug_staged_home_experts = 0
        self._debug_unique_remote_sources = 0
        self._debug_duplicate_remote_reads_avoided = 0
        self._debug_home_weight_barriers = 0
        self._debug_spare_grad_barriers = 0
        self._debug_backward_schedule_entries = 0
        self._debug_workspace_sync_mode = "double_buffered_stage_barrier"
        self._workspaces: dict[
            tuple[tuple[int, ...], torch.dtype, str, torch.dtype],
            _SymmetricExpertWeightDispatchWorkspace,
        ] = {}

    @staticmethod
    def availability_error() -> Optional[str]:
        """Return why SymmMem dispatch is unavailable, or None when usable."""

        try:
            _get_symmetric_memory_module()
        except RuntimeError as exc:
            return str(exc)
        return None

    def _reset_debug_counters(self) -> None:
        self._debug_low_level_get_calls = 0
        self._debug_staged_home_experts = 0
        self._debug_unique_remote_sources = 0
        self._debug_duplicate_remote_reads_avoided = 0
        self._debug_home_weight_barriers = 0
        self._debug_spare_grad_barriers = 0
        self._debug_backward_schedule_entries = 0
        self._debug_workspace_sync_mode = "double_buffered_stage_barrier"

    def _validate_weights(self, local_home_weights: tuple[torch.Tensor, ...]) -> None:
        if len(local_home_weights) != self.num_local_home_experts:
            raise ValueError(
                f"Expected {self.num_local_home_experts} local home weights, "
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
            if weight.dtype != reference_weight.dtype:
                raise ValueError(
                    "All expert weights in one dispatch call must have the same dtype; "
                    f"got {weight.dtype} and {reference_weight.dtype}."
                )
            if weight.device != reference_weight.device:
                raise ValueError(
                    "All expert weights in one dispatch call must be on the same device; "
                    f"got {weight.device} and {reference_weight.device}."
                )

    def _get_workspace(
        self,
        reference_weight: torch.Tensor,
        *,
        grad_combine_dtype: Optional[torch.dtype] = None,
        reference_weight_dtype: Optional[torch.dtype] = None,
    ) -> _SymmetricExpertWeightDispatchWorkspace:
        weight_dtype = reference_weight_dtype or reference_weight.dtype
        if grad_combine_dtype is None:
            grad_combine_dtype = self._grad_combine_dtype(weight_dtype)
        key = (
            tuple(reference_weight.shape),
            weight_dtype,
            str(reference_weight.device),
            grad_combine_dtype,
        )
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace

        weight_numel = reference_weight.numel()
        device = reference_weight.device
        home_weight_buffer = self._symm_mem.empty(
            _SYMMETRIC_MEMORY_BUFFER_SLOTS * self.num_local_home_experts * weight_numel,
            dtype=weight_dtype,
            device=device,
        )
        spare_grad_buffer = self._symm_mem.empty(
            _SYMMETRIC_MEMORY_BUFFER_SLOTS * self.num_local_spare_experts * weight_numel,
            dtype=grad_combine_dtype,
            device=device,
        )
        workspace = _SymmetricExpertWeightDispatchWorkspace(
            home_weight_buffer=home_weight_buffer,
            spare_grad_buffer=spare_grad_buffer,
            home_weight_handle=_rendezvous_symmetric_buffer(
                self._symm_mem, home_weight_buffer, self.ep_group
            ),
            spare_grad_handle=_rendezvous_symmetric_buffer(
                self._symm_mem, spare_grad_buffer, self.ep_group
            ),
        )
        self._workspaces[key] = workspace
        return workspace

    def _next_home_weight_slot(
        self, workspace: _SymmetricExpertWeightDispatchWorkspace
    ) -> int:
        slot = workspace.next_home_weight_slot
        workspace.next_home_weight_slot = (slot + 1) % _SYMMETRIC_MEMORY_BUFFER_SLOTS
        return slot

    def _next_spare_grad_slot(
        self, workspace: _SymmetricExpertWeightDispatchWorkspace
    ) -> int:
        slot = workspace.next_spare_grad_slot
        workspace.next_spare_grad_slot = (slot + 1) % _SYMMETRIC_MEMORY_BUFFER_SLOTS
        return slot

    def _copy_from_symmetric_peer(
        self, dst_flat: torch.Tensor, handle, *, peer: int, offset: int
    ) -> None:
        self._debug_low_level_get_calls += 1
        if hasattr(self._symm_mem, "get"):
            self._symm_mem.get(dst_flat, handle, peer=peer, offset=offset)
            return

        if not hasattr(handle, "get_buffer"):
            raise RuntimeError(
                "Symmetric Memory handle does not expose get_buffer, and "
                "torch.distributed._symmetric_memory.get is unavailable."
            )
        peer_buffer = handle.get_buffer(peer, (offset + dst_flat.numel(),), dtype=dst_flat.dtype)
        dst_flat.copy_(peer_buffer[offset : offset + dst_flat.numel()])

    def dispatch(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *expert_weights: torch.Tensor,
        runtime_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        if not expert_weights:
            raise ValueError("At least one local home weight is required.")
        outputs = _SymmetricMemoryExpertWeightDispatch.apply(
            self,
            metadata,
            runtime_weight_dtype,
            self._grad_combine_dtype(expert_weights[0].dtype),
            *expert_weights,
        )
        return list(outputs)

    def fold_spare_gradients(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *local_spare_grads: torch.Tensor,
        reference_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        """Fold local spare-slot gradients through low-level SymmMem reads."""

        if not local_spare_grads:
            raise ValueError("At least one local spare gradient is required.")
        reference_grad = local_spare_grads[0]
        effective_weight_dtype = reference_weight_dtype or reference_grad.dtype
        grad_combine_dtype = self._grad_combine_dtype(effective_weight_dtype)
        workspace = self._get_workspace(
            reference_grad,
            grad_combine_dtype=grad_combine_dtype,
            reference_weight_dtype=effective_weight_dtype,
        )
        spare_grad_slot = self._next_spare_grad_slot(workspace)
        spare_grad_view = workspace.spare_grad_buffer.view(
            _SYMMETRIC_MEMORY_BUFFER_SLOTS,
            self.num_local_spare_experts,
            *reference_grad.shape,
        )[spare_grad_slot]

        with torch.no_grad():
            spare_grad_view.zero_()
            for local_spare_idx in metadata.recv_local_spare_indices:
                spare_grad_view[local_spare_idx].copy_(
                    local_spare_grads[local_spare_idx].detach().to(dtype=grad_combine_dtype)
                )

        workspace.spare_grad_handle.barrier()
        self._debug_spare_grad_barriers += 1

        local_home_grads = [
            torch.zeros(reference_grad.shape, dtype=grad_combine_dtype, device=reference_grad.device)
            for _ in range(self.num_local_home_experts)
        ]
        for (
            dest_rank,
            dest_local_spare,
            source_local_home,
            dest_spare_offset,
        ) in metadata.backward_grad_sources:
            if dest_rank == self.ep_rank:
                local_home_grads[source_local_home].add_(spare_grad_view[dest_local_spare])
            else:
                tmp_grad = torch.empty(
                    reference_grad.shape, dtype=grad_combine_dtype, device=reference_grad.device
                )
                offset = (
                    spare_grad_slot * self.num_local_spare_experts + dest_spare_offset
                ) * reference_grad.numel()
                self._copy_from_symmetric_peer(
                    tmp_grad.contiguous().view(-1),
                    workspace.spare_grad_handle,
                    peer=dest_rank,
                    offset=offset,
                )
                local_home_grads[source_local_home].add_(tmp_grad)
        return local_home_grads


class _HybridEPExpertWeightChunkDispatch(torch.autograd.Function):
    """Autograd bridge for HybridEP weight-chunk dispatch.

    The generic HybridEP token wrapper propagates routing probability gradients.
    Expert weights do not have routing probabilities, so this bridge matches the
    historical Echo expert-weight path and calls the HybridEP primitives with
    ``probs=None`` in both forward and backward.
    """

    _buffer = None
    _buffer_key = None

    @staticmethod
    def _get_buffer(
        group: torch.distributed.ProcessGroup,
        hidden_dim: int,
        max_num_tokens: int,
        num_local_experts: int,
        num_sms_dispatch_api: Optional[int],
        num_sms_combine_api: Optional[int],
        num_blocks_permute: Optional[int],
        num_blocks_unpermute: Optional[int],
        num_sms_preprocessing_api: Optional[int],
    ):
        if HybridEPBuffer is None:
            raise RuntimeError("HybridEP is not installed.")

        key = (
            group,
            hidden_dim,
            max_num_tokens,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_blocks_permute,
            num_blocks_unpermute,
            num_sms_preprocessing_api,
        )
        if _HybridEPExpertWeightChunkDispatch._buffer is not None:
            current_key = _HybridEPExpertWeightChunkDispatch._buffer_key
            if (
                current_key is not None
                and current_key[0] is group
                and current_key[1] == hidden_dim
                and current_key[2] >= max_num_tokens
                and current_key[3:] == key[3:]
            ):
                return _HybridEPExpertWeightChunkDispatch._buffer

        kwargs = {}
        if num_sms_dispatch_api is not None:
            kwargs["num_sms_dispatch_api"] = num_sms_dispatch_api
        if num_sms_combine_api is not None:
            kwargs["num_sms_combine_api"] = num_sms_combine_api
        if num_blocks_permute is not None:
            kwargs["num_blocks_permute"] = num_blocks_permute
        if num_blocks_unpermute is not None:
            kwargs["num_blocks_unpermute"] = num_blocks_unpermute
        if num_sms_preprocessing_api is not None:
            kwargs["num_sms_preprocessing_api"] = num_sms_preprocessing_api

        _HybridEPExpertWeightChunkDispatch._buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_tokens,
            num_local_experts=num_local_experts,
            use_fp8=False,
            **kwargs,
        )
        _HybridEPExpertWeightChunkDispatch._buffer_key = key
        return _HybridEPExpertWeightChunkDispatch._buffer

    @staticmethod
    def forward(
        ctx,
        flat_chunks: torch.Tensor,
        routing_map: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_permuted_tokens: int,
        num_sms_dispatch_api: Optional[int],
        num_sms_combine_api: Optional[int],
        num_blocks_permute: Optional[int],
        num_blocks_unpermute: Optional[int],
        num_sms_preprocessing_api: Optional[int],
    ) -> torch.Tensor:
        """Dispatch flattened expert-weight chunks through HybridEP."""

        buffer = _HybridEPExpertWeightChunkDispatch._get_buffer(
            group,
            flat_chunks.shape[1],
            flat_chunks.shape[0],
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_blocks_permute,
            num_blocks_unpermute,
            num_sms_preprocessing_api,
        )
        dispatched_chunks, _, _, _, handle = buffer.dispatch_with_permute(
            hidden=flat_chunks,
            routing_map=routing_map,
            probs=None,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=None,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )

        ctx.handle = handle
        return dispatched_chunks

    @staticmethod
    def backward(ctx, grad_dispatched_chunks: torch.Tensor):
        """Combine HybridEP-dispatched chunk gradients back to input chunks."""

        buffer = _HybridEPExpertWeightChunkDispatch._buffer
        if buffer is None:
            raise RuntimeError("HybridEP expert-weight dispatch buffer is not initialized.")

        combined_chunks, _ = buffer.combine_with_unpermute(
            hidden=grad_dispatched_chunks.contiguous(),
            probs=None,
            handle=ctx.handle,
            pad_multiple=None,
        )
        return combined_chunks, None, None, None, None, None, None, None, None, None


class HybridEPExpertWeightDispatcher(AllToAllExpertWeightDispatcher):
    """Experimental HybridEP expert-weight dispatcher.

    This backend treats flattened expert-weight chunks as HybridEP token rows and
    routes those rows to spare expert slots. HybridEP's autograd combine path
    folds spare gradients back to the home weight chunks.
    """

    @staticmethod
    def availability_error() -> Optional[str]:
        """Return why HybridEP dispatch is unavailable, or None when usable."""

        if not HAVE_HYBRIDEP or HybridEPBuffer is None:
            return (
                "HybridEP is not installed. Please install the DeepEP HybridEP package "
                "or use moe_balance_expert_weight_dispatch_backend='all_to_all'."
            )
        return None

    @staticmethod
    def _weight_chunk_size(reference_weight: torch.Tensor) -> int:
        return 8192

    def _validate_weights(self, local_home_weights: tuple[torch.Tensor, ...]) -> None:
        if len(local_home_weights) != self.num_local_home_experts:
            raise ValueError(
                f"Expected {self.num_local_home_experts} local home weights, "
                f"got {len(local_home_weights)}."
            )
        if not local_home_weights:
            raise ValueError("At least one local home weight is required.")

        reference_weight = local_home_weights[0]
        if reference_weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "HybridEP expert-weight dispatch supports only 16-bit floating "
                f"weights (fp16/bf16); got {reference_weight.dtype}."
            )
        for weight in local_home_weights:
            if weight.shape != reference_weight.shape:
                raise ValueError(
                    "All expert weights in one dispatch call must have the same shape; "
                    f"got {tuple(weight.shape)} and {tuple(reference_weight.shape)}."
                )
            if weight.dtype != reference_weight.dtype:
                raise ValueError(
                    "All expert weights in one dispatch call must have the same dtype; "
                    f"got {weight.dtype} and {reference_weight.dtype}."
                )
            if weight.device != reference_weight.device:
                raise ValueError(
                    "All expert weights in one dispatch call must be on the same device; "
                    f"got {weight.device} and {reference_weight.device}."
                )

    def _chunk_local_home_weights(
        self, local_home_weights: tuple[torch.Tensor, ...]
    ) -> tuple[list[torch.Tensor], int, int, int]:
        reference_weight = local_home_weights[0]
        weight_numel = reference_weight.numel()
        chunk_size = self._weight_chunk_size(reference_weight)
        chunks_per_weight = math.ceil(weight_numel / chunk_size)
        padded_numel = chunks_per_weight * chunk_size

        chunked_weights = []
        for weight in local_home_weights:
            flat_weight = weight.reshape(-1)
            if padded_numel != weight_numel:
                flat_weight = torch.cat(
                    [flat_weight, flat_weight.new_zeros(padded_numel - weight_numel)], dim=0
                )
            chunked_weights.append(flat_weight.view(chunks_per_weight, chunk_size))
        return chunked_weights, chunks_per_weight, chunk_size, weight_numel

    def _flatten_local_home_weights(
        self, local_home_weights: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, int, int]:
        chunked_weights, chunks_per_weight, chunk_size, weight_numel = (
            self._chunk_local_home_weights(local_home_weights)
        )
        if not chunked_weights:
            reference_weight = local_home_weights[0]
            return reference_weight.new_empty((0, chunk_size)), chunks_per_weight, weight_numel
        return torch.cat(chunked_weights, dim=0).contiguous(), chunks_per_weight, weight_numel

    def _make_local_spare_sources(
        self, metadata: ExpertWeightDispatchMetadata
    ) -> list[Optional[tuple[int, int]]]:
        local_spare_home_indices = metadata.local_spare_home_indices.cpu().tolist()
        sources: list[Optional[tuple[int, int]]] = [None] * self.num_local_spare_experts
        for local_spare_idx, home_idx in enumerate(local_spare_home_indices):
            if home_idx < 0:
                continue
            sources[local_spare_idx] = (
                home_idx // self.num_local_home_experts,
                home_idx % self.num_local_home_experts,
            )
        return sources

    def _make_remote_spare_aliases(
        self, local_spare_sources: list[Optional[tuple[int, int]]]
    ) -> list[Optional[int]]:
        aliases: list[Optional[int]] = [None] * self.num_local_spare_experts
        for local_spare_idx, source in enumerate(local_spare_sources):
            if source is None:
                continue
            source_rank, local_home_idx = source
            if source_rank == self.ep_rank:
                continue
            aliases[local_spare_idx] = min(
                idx
                for idx, candidate_source in enumerate(local_spare_sources)
                if candidate_source == (source_rank, local_home_idx)
            )
        return aliases

    def _flatten_coalesced_route_chunks(
        self, metadata: ExpertWeightDispatchMetadata, local_home_weights: tuple[torch.Tensor, ...]
    ) -> tuple[
        torch.Tensor, torch.Tensor, int, int, list[Optional[tuple[int, int]]], list[Optional[int]]
    ]:
        reference_weight = local_home_weights[0]
        chunked_weights, chunks_per_weight, chunk_size, weight_numel = (
            self._chunk_local_home_weights(local_home_weights)
        )
        local_home_routes = metadata.local_to_global_routing_map.reshape(
            self.num_local_home_experts, self.ep_size, self.num_local_spare_experts
        )

        pair_chunks: list[torch.Tensor] = []
        pair_routes: list[torch.Tensor] = []
        for local_home_idx in range(self.num_local_home_experts):
            for dest_rank in range(self.ep_size):
                if dest_rank == self.ep_rank:
                    continue
                local_spare_indices = torch.where(local_home_routes[local_home_idx, dest_rank])[0]
                if local_spare_indices.numel() == 0:
                    continue
                representative_local_spare_idx = int(local_spare_indices.min().item())
                representative_global_spare_idx = (
                    dest_rank * self.num_local_spare_experts + representative_local_spare_idx
                )
                route = torch.zeros(
                    (chunks_per_weight, self.num_spare_experts),
                    dtype=torch.bool,
                    device=reference_weight.device,
                )
                route[:, representative_global_spare_idx] = True
                pair_chunks.append(chunked_weights[local_home_idx])
                pair_routes.append(route)

        if pair_chunks:
            flat_chunks = torch.cat(pair_chunks, dim=0).contiguous()
            chunk_routing_map = torch.cat(pair_routes, dim=0).contiguous()
        else:
            flat_chunks = reference_weight.new_empty((0, chunk_size))
            chunk_routing_map = torch.zeros(
                (0, self.num_spare_experts), dtype=torch.bool, device=reference_weight.device
            )
        local_spare_sources = self._make_local_spare_sources(metadata)
        return (
            flat_chunks,
            chunk_routing_map,
            chunks_per_weight,
            weight_numel,
            local_spare_sources,
            self._make_remote_spare_aliases(local_spare_sources),
        )

    def _pad_chunks_for_hybridep(
        self, flat_chunks: torch.Tensor, chunk_routing_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_rows = torch.tensor(
            [flat_chunks.shape[0]], dtype=torch.long, device=flat_chunks.device
        )
        max_rows = local_rows.clone()
        torch.distributed.all_reduce(
            max_rows, op=torch.distributed.ReduceOp.MAX, group=self.ep_group
        )
        padded_rows = int(max_rows.item())
        if padded_rows == 0:
            return flat_chunks, chunk_routing_map
        padded_rows += -padded_rows % _HYBRIDEP_WEIGHT_TOKEN_ALIGNMENT
        if padded_rows == flat_chunks.shape[0]:
            return flat_chunks, chunk_routing_map

        pad_rows = padded_rows - flat_chunks.shape[0]
        if pad_rows < 0:
            raise RuntimeError("HybridEP input padding computed a negative pad row count.")
        flat_chunks = torch.cat(
            [flat_chunks, flat_chunks.new_zeros((pad_rows, flat_chunks.shape[1]))], dim=0
        )
        chunk_routing_map = torch.cat(
            [
                chunk_routing_map,
                chunk_routing_map.new_zeros((pad_rows, chunk_routing_map.shape[1])),
            ],
            dim=0,
        )
        return flat_chunks.contiguous(), chunk_routing_map.contiguous()

    def dispatch(
        self,
        metadata: ExpertWeightDispatchMetadata,
        *expert_weights: torch.Tensor,
        runtime_weight_dtype: Optional[torch.dtype] = None,
    ) -> list[torch.Tensor]:
        availability_error = self.availability_error()
        if availability_error is not None:
            raise RuntimeError(availability_error)
        self._validate_weights(expert_weights)

        reference_weight = expert_weights[0]
        if runtime_weight_dtype is not None and runtime_weight_dtype != reference_weight.dtype:
            raise ValueError(
                "HybridEP expert-weight dispatch does not support overriding "
                "runtime_weight_dtype."
            )
        zero_home_dependency = reference_weight.new_zeros(())
        for weight in expert_weights:
            zero_home_dependency = zero_home_dependency + weight.sum() * 0.0
        (
            flat_chunks,
            chunk_routing_map,
            chunks_per_weight,
            weight_numel,
            local_spare_sources,
            remote_spare_aliases,
        ) = self._flatten_coalesced_route_chunks(metadata, expert_weights)
        flat_chunks, chunk_routing_map = self._pad_chunks_for_hybridep(
            flat_chunks, chunk_routing_map
        )

        num_local_output_chunks = self.num_local_spare_experts * chunks_per_weight
        if flat_chunks.shape[0] == 0:
            outputs: list[torch.Tensor] = []
            for source in local_spare_sources:
                if source is None:
                    outputs.append(
                        reference_weight.new_zeros(reference_weight.shape) + zero_home_dependency
                    )
                    continue
                source_rank, local_home_idx = source
                if source_rank != self.ep_rank:
                    raise RuntimeError(
                        "HybridEP expert-weight dispatch has remote receive aliases but no "
                        "remote send rows in the EP group."
                    )
                outputs.append(expert_weights[local_home_idx] + zero_home_dependency)
            return outputs

        flat_chunks = flat_chunks + zero_home_dependency
        dispatched_chunks = _HybridEPExpertWeightChunkDispatch.apply(
            flat_chunks,
            chunk_routing_map,
            self.ep_group,
            self.num_local_spare_experts,
            num_local_output_chunks,
            getattr(self.config, "moe_hybridep_num_sms", None),
            getattr(self.config, "moe_hybridep_num_sms", None),
            getattr(self.config, "moe_hybridep_num_blocks_permute", None),
            getattr(self.config, "moe_hybridep_num_blocks_unpermute", None),
            getattr(self.config, "moe_hybridep_num_sms_preprocessing", 108),
        )

        if dispatched_chunks.shape[0] != num_local_output_chunks:
            raise RuntimeError(
                "HybridEP expert-weight dispatch returned an unexpected number of chunks: "
                f"expected {num_local_output_chunks}, got {dispatched_chunks.shape[0]}."
            )
        dispatched_slot_chunks = dispatched_chunks.chunk(self.num_local_spare_experts, dim=0)
        zero_dispatch_dependency = dispatched_chunks.masked_fill(
            torch.ones((), dtype=torch.bool, device=dispatched_chunks.device), 0.0
        ).sum()
        zero_dependency = zero_home_dependency + zero_dispatch_dependency
        representative_outputs: dict[int, torch.Tensor] = {}
        outputs: list[torch.Tensor] = []
        for local_spare_idx, source in enumerate(local_spare_sources):
            if source is None:
                outputs.append(reference_weight.new_zeros(reference_weight.shape) + zero_dependency)
                continue

            source_rank, local_home_idx = source
            if source_rank == self.ep_rank:
                outputs.append(expert_weights[local_home_idx] + zero_dependency)
                continue

            representative_local_spare_idx = remote_spare_aliases[local_spare_idx]
            if representative_local_spare_idx is None:
                raise RuntimeError(
                    "HybridEP expert-weight dispatch missing a representative slot for "
                    f"remote local spare {local_spare_idx}."
                )
            if representative_local_spare_idx not in representative_outputs:
                flat_weight = dispatched_slot_chunks[representative_local_spare_idx].reshape(-1)[
                    :weight_numel
                ]
                representative_outputs[representative_local_spare_idx] = (
                    flat_weight.view(reference_weight.shape) + zero_dependency
                )
            outputs.append(representative_outputs[representative_local_spare_idx])
        return outputs
