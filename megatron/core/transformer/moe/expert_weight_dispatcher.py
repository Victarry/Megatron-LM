# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert-weight dispatch for BalancedMoELayer.

The dispatcher moves checkpoint-owned home expert weights into runtime spare
slots. Forward sends only active home-to-spare pairs across the EP group.
Backward explicitly folds spare gradients back to the owning home expert
parameters through the reverse communication pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    send_local_home_indices: list[int]
    recv_local_spare_indices: list[int]


@dataclass
class _SymmetricExpertWeightDispatchWorkspace:
    """Symmetric-memory buffers cached for one weight shape/dtype/device."""

    home_weight_buffer: torch.Tensor
    spare_grad_buffer: torch.Tensor
    home_weight_handle: Any
    spare_grad_handle: Any


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
            reference_weight.new_zeros(reference_weight.shape)
            for _ in range(num_local_spare_experts)
        ]
        for recv_offset, local_spare_idx in enumerate(recv_local_spare_indices):
            outputs[local_spare_idx] = recv_tensor[recv_offset].clone()

        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size
        ctx.num_local_home_experts = num_local_home_experts
        ctx.weight_shape = tuple(reference_weight.shape)
        ctx.weight_dtype = reference_weight.dtype
        ctx.weight_device = reference_weight.device
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.send_local_home_indices = send_local_home_indices
        ctx.recv_local_spare_indices = recv_local_spare_indices

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        grad_send_tensors: list[torch.Tensor] = []
        for local_spare_idx in ctx.recv_local_spare_indices:
            grad_output = grad_outputs[local_spare_idx]
            if grad_output is None:
                grad_send_tensors.append(
                    torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=ctx.weight_device)
                )
            else:
                grad_send_tensors.append(grad_output)

        if grad_send_tensors:
            grad_send_tensor = torch.stack(grad_send_tensors, dim=0).contiguous()
        else:
            grad_send_tensor = torch.empty(
                (0, *ctx.weight_shape), dtype=ctx.weight_dtype, device=ctx.weight_device
            )
        grad_recv_tensor = torch.empty(
            (sum(ctx.input_splits), *ctx.weight_shape),
            dtype=ctx.weight_dtype,
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
            torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=ctx.weight_device)
            for _ in range(ctx.num_local_home_experts)
        ]
        for local_home_idx, grad in zip(ctx.send_local_home_indices, grad_recv_tensor):
            local_home_grads[local_home_idx].add_(grad)

        return (None, None, None, None, None, None, None, None, None, *local_home_grads)


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

        send_local_home_indices_by_rank: list[list[int]] = [[] for _ in range(self.ep_size)]
        recv_local_spare_indices_by_rank: list[list[int]] = [[] for _ in range(self.ep_size)]
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
            if dest_rank == self.ep_rank:
                recv_local_spare_indices_by_rank[source_rank].append(local_spare_idx)

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
        )

    def dispatch(
        self, metadata: ExpertWeightDispatchMetadata, *expert_weights: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return weights for this rank's local spare slots."""

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
            *expert_weights,
        )
        return list(outputs)

    def expert_dispatch(
        self, metadata: ExpertWeightDispatchMetadata, *expert_weights: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compatibility alias for the old Echo dispatcher method name."""

        return self.dispatch(metadata, *expert_weights)


def _get_symmetric_memory_module():
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as exc:
        raise RuntimeError("torch.distributed._symmetric_memory is not importable.") from exc

    missing = [name for name in ("empty", "rendezvous") if not hasattr(symm_mem, name)]
    if missing:
        raise RuntimeError(
            "torch.distributed._symmetric_memory is missing required API(s): "
            + ", ".join(missing)
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
        *local_home_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        dispatcher._validate_weights(local_home_weights)
        reference_weight = local_home_weights[0]
        weight_numel = reference_weight.numel()
        workspace = dispatcher._get_workspace(reference_weight)
        home_weight_view = workspace.home_weight_buffer.view(
            dispatcher.num_local_home_experts, *reference_weight.shape
        )

        with torch.no_grad():
            for local_home_idx, weight in enumerate(local_home_weights):
                home_weight_view[local_home_idx].copy_(weight.detach())

        workspace.home_weight_handle.barrier()

        outputs: list[torch.Tensor] = [
            reference_weight.new_zeros(reference_weight.shape)
            for _ in range(dispatcher.num_local_spare_experts)
        ]
        for local_spare_idx in range(dispatcher.num_local_spare_experts):
            home_idx = int(metadata.local_spare_home_indices[local_spare_idx].item())
            if home_idx < 0:
                continue

            source_rank = home_idx // dispatcher.num_local_home_experts
            source_local_home = home_idx % dispatcher.num_local_home_experts
            output = torch.empty_like(reference_weight)
            if source_rank == dispatcher.ep_rank:
                output.copy_(home_weight_view[source_local_home])
            else:
                offset = source_local_home * weight_numel
                dispatcher._copy_from_symmetric_peer(
                    output.contiguous().view(-1),
                    workspace.home_weight_handle,
                    peer=source_rank,
                    offset=offset,
                )
            outputs[local_spare_idx] = output

        workspace.home_weight_handle.barrier()

        ctx.dispatcher = dispatcher
        ctx.dispatch_metadata = metadata
        ctx.workspace = workspace
        ctx.weight_shape = tuple(reference_weight.shape)
        ctx.weight_dtype = reference_weight.dtype
        ctx.weight_device = reference_weight.device
        ctx.weight_numel = weight_numel

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: Optional[torch.Tensor]):
        dispatcher: SymmetricMemoryExpertWeightDispatcher = ctx.dispatcher
        metadata: ExpertWeightDispatchMetadata = ctx.dispatch_metadata
        workspace: _SymmetricExpertWeightDispatchWorkspace = ctx.workspace

        spare_grad_view = workspace.spare_grad_buffer.view(
            dispatcher.num_local_spare_experts, *ctx.weight_shape
        )
        with torch.no_grad():
            spare_grad_view.zero_()
            for local_spare_idx in metadata.recv_local_spare_indices:
                grad_output = grad_outputs[local_spare_idx]
                if grad_output is not None:
                    spare_grad_view[local_spare_idx].copy_(grad_output.detach())

        workspace.spare_grad_handle.barrier()

        local_home_grads = [
            torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=ctx.weight_device)
            for _ in range(dispatcher.num_local_home_experts)
        ]

        for spare_idx in range(dispatcher.num_spare_experts):
            home_indices = torch.where(metadata.global_routing_map[:, spare_idx])[0]
            if home_indices.numel() == 0:
                continue
            home_idx = int(home_indices.item())
            source_rank = home_idx // dispatcher.num_local_home_experts
            if source_rank != dispatcher.ep_rank:
                continue

            source_local_home = home_idx % dispatcher.num_local_home_experts
            dest_rank = spare_idx // dispatcher.num_local_spare_experts
            dest_local_spare = spare_idx % dispatcher.num_local_spare_experts
            if dest_rank == dispatcher.ep_rank:
                local_home_grads[source_local_home].add_(spare_grad_view[dest_local_spare])
            else:
                tmp_grad = torch.empty(
                    ctx.weight_shape, dtype=ctx.weight_dtype, device=ctx.weight_device
                )
                offset = dest_local_spare * ctx.weight_numel
                dispatcher._copy_from_symmetric_peer(
                    tmp_grad.contiguous().view(-1),
                    workspace.spare_grad_handle,
                    peer=dest_rank,
                    offset=offset,
                )
                local_home_grads[source_local_home].add_(tmp_grad)

        workspace.spare_grad_handle.barrier()

        return (None, None, *local_home_grads)


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
        self._workspaces: dict[
            tuple[tuple[int, ...], torch.dtype, str], _SymmetricExpertWeightDispatchWorkspace
        ] = {}

    @staticmethod
    def availability_error() -> Optional[str]:
        try:
            _get_symmetric_memory_module()
        except RuntimeError as exc:
            return str(exc)
        return None

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
        self, reference_weight: torch.Tensor
    ) -> _SymmetricExpertWeightDispatchWorkspace:
        key = (
            tuple(reference_weight.shape),
            reference_weight.dtype,
            str(reference_weight.device),
        )
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace

        weight_numel = reference_weight.numel()
        device = reference_weight.device
        dtype = reference_weight.dtype
        home_weight_buffer = self._symm_mem.empty(
            self.num_local_home_experts * weight_numel,
            dtype=dtype,
            device=device,
        )
        spare_grad_buffer = self._symm_mem.empty(
            self.num_local_spare_experts * weight_numel,
            dtype=dtype,
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

    def _copy_from_symmetric_peer(
        self,
        dst_flat: torch.Tensor,
        handle,
        *,
        peer: int,
        offset: int,
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
        peer_buffer = handle.get_buffer(
            peer, (offset + dst_flat.numel(),), dtype=dst_flat.dtype
        )
        dst_flat.copy_(peer_buffer[offset : offset + dst_flat.numel()])

    def dispatch(
        self, metadata: ExpertWeightDispatchMetadata, *expert_weights: torch.Tensor
    ) -> list[torch.Tensor]:
        outputs = _SymmetricMemoryExpertWeightDispatch.apply(self, metadata, *expert_weights)
        return list(outputs)
