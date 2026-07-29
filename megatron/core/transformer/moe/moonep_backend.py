# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP data-plane integration for :class:`BalancedMoELayer`."""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.utils import is_te_min_version

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from megatron.core.transformer.transformer_config import TransformerConfig


def _import_moonep():
    try:
        import moonep
    except ImportError as exc:
        raise RuntimeError(
            "moe_balance_backend='moonep' requires the pinned Victarry/MoonEP fork "
            "and its CUDA extension to be installed"
        ) from exc
    return moonep


class _DenseRouteToMoonEP(torch.autograd.Function):
    """Convert MCore's dense route representation while preserving probability grads."""

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if probs.ndim != 2 or routing_map.shape != probs.shape:
            raise ValueError(
                "MoonEP route adapter requires probs and routing_map with matching [S, E] shapes"
            )
        if probs.dtype != torch.float32:
            raise ValueError(f"MoonEP route probabilities must be FP32, got {probs.dtype}")
        if routing_map.dtype != torch.bool:
            raise ValueError(f"MoonEP routing_map must be bool, got {routing_map.dtype}")
        if not 1 <= topk <= probs.shape[1]:
            raise ValueError(f"MoonEP topk must be in [1, E], got {topk}")

        route_counts = routing_map.sum(dim=-1)
        if torch.any(route_counts > topk):
            raise ValueError("MoonEP routing_map contains more than topk routes for a token")

        num_tokens, num_experts = probs.shape
        expert_range = torch.arange(num_experts, device=probs.device, dtype=torch.int64)
        dense_ids = expert_range.expand(num_tokens, -1)
        selected_ids = torch.where(routing_map, dense_ids, num_experts)
        expert_ids = selected_ids.sort(dim=-1).values[:, :topk]
        valid = expert_ids != num_experts

        # Padding tokens have no valid route. Fill those physical slots evenly so
        # they do not manufacture a planner hotspot; their route weights stay zero.
        token_ids = torch.arange(num_tokens, device=probs.device, dtype=torch.int64)[:, None]
        slot_ids = torch.arange(topk, device=probs.device, dtype=torch.int64)[None, :]
        padding_ids = (token_ids * topk + slot_ids) % num_experts
        expert_ids = torch.where(valid, expert_ids, padding_ids)
        route_weights = probs.gather(1, expert_ids).masked_fill(~valid, 0.0).contiguous()
        expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
        tokens_per_expert = torch.bincount(
            expert_ids.reshape(-1), minlength=num_experts
        ).to(torch.int32)

        ctx.save_for_backward(expert_ids, valid)
        ctx.probs_shape = probs.shape
        return route_weights, expert_ids_i32, tokens_per_expert

    @staticmethod
    def backward(ctx, grad_route_weights, _grad_expert_ids, _grad_tokens_per_expert):
        expert_ids, valid = ctx.saved_tensors
        grad_probs = None
        if grad_route_weights is not None:
            grad_probs = grad_route_weights.new_zeros(ctx.probs_shape)
            grad_probs.scatter_add_(
                1, expert_ids, grad_route_weights.masked_fill(~valid, 0.0)
            )
        return grad_probs, None, None


def adapt_mcore_route(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MoonEP ``route_weights``, ``expert_ids`` and local expert counts."""

    return _DenseRouteToMoonEP.apply(probs, routing_map, topk)


@dataclass(frozen=True)
class MoonEPRuntimeKey:
    group_id: int
    device: int
    num_tokens: int
    hidden_size: int
    topk: int
    num_experts: int
    num_spare_experts: int
    num_sms: int | None
    token_padding: int
    comm_stream_priority: int
    enable_pdl: bool


class MoonEPRuntime:
    """Own one reusable MoonEP communication buffer for a static token shape."""

    def __init__(
        self,
        key: MoonEPRuntimeKey,
        group: ProcessGroup,
        ep_size: int,
    ) -> None:
        moonep = _import_moonep()
        moonep.validate_runtime_environment(group)
        self.key = key
        self.group = group
        self.ep_size = ep_size
        self.buffer = moonep.Buffer(
            S=key.num_tokens,
            H=key.hidden_size,
            K=key.topk,
            E=key.num_experts,
            B=key.num_spare_experts,
            num_ep_ranks=ep_size,
            num_sms=key.num_sms,
            token_padding=key.token_padding,
            group=group,
            comm_stream_priority=key.comm_stream_priority,
            enable_pdl=key.enable_pdl,
            explicitly_destroy=True,
        )
        self._closed = False

    @staticmethod
    def _wait(event) -> None:
        if event is not None:
            event.wait(torch.cuda.current_stream())

    def dispatch_and_prefetch(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        full_weights: Sequence[torch.Tensor],
    ):
        # MoonEP staging uses ordinary PyTorch copy_ operations around its
        # kernels. Keep those implementation details out of autograd; the
        # explicit dispatch/combine Functions below define the true gradient.
        with torch.no_grad():
            dispatched_hidden, dispatched_weights, cu_seqlens, plan, _ = (
                self.buffer.dispatch(
                    hidden,
                    route_weights,
                    expert_ids,
                    tokens_per_expert,
                    async_finish=True,
                    zero_copy=False,
                )
            )
            event = self.buffer.prefetch_weights(
                plan, full_weights, async_finish=True
            )
        self._wait(event)
        return dispatched_hidden, dispatched_weights, cu_seqlens, plan

    def combine(
        self,
        plan,
        hidden: torch.Tensor,
        route_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            output, output_weights, event = self.buffer.combine(
                plan=plan,
                hidden_nvsh=hidden,
                route_weights_nvs=route_weights,
                async_finish=True,
                zero_copy=False,
            )
        self._wait(event)
        return output, output_weights

    def redispatch(self, plan, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dispatched, _, _, _, event = self.buffer.dispatch(
                hidden,
                plan=plan,
                async_finish=True,
                zero_copy=False,
            )
        self._wait(event)
        return dispatched

    def reduce_grads(self, plan, table) -> None:
        with torch.no_grad():
            event = self.buffer.reduce_grads(
                plan,
                table.full_grads,
                table.reduce_buffers,
                async_finish=True,
            )
        self._wait(event)

    def close(self) -> None:
        if self._closed:
            return
        self.buffer.destroy()
        self._closed = True


class MoonEPRuntimeRegistry:
    """Process-local owner for MoonEP buffers and VMM expert tables."""

    _lock = threading.Lock()
    _runtimes: dict[MoonEPRuntimeKey, MoonEPRuntime] = {}
    _resources: list[object] = []

    @classmethod
    def get(
        cls,
        *,
        config: TransformerConfig,
        group: ProcessGroup,
        ep_size: int,
        num_tokens: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        num_spare_experts: int,
    ) -> MoonEPRuntime:
        key = MoonEPRuntimeKey(
            group_id=id(group),
            device=torch.cuda.current_device(),
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            num_spare_experts=num_spare_experts,
            num_sms=config.moe_balance_moonep_num_sms,
            token_padding=config.moe_balance_moonep_token_padding,
            comm_stream_priority=config.moe_balance_moonep_comm_stream_priority,
            enable_pdl=config.moe_balance_moonep_enable_pdl,
        )
        with cls._lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                runtime = MoonEPRuntime(key, group, ep_size)
                cls._runtimes[key] = runtime
            return runtime

    @classmethod
    def register_resource(cls, resource) -> None:
        with cls._lock:
            cls._resources.append(resource)

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            resources = cls._resources
            runtimes = list(cls._runtimes.values())
            cls._resources = []
            cls._runtimes = {}
        for resource in resources:
            resource.close()
        for runtime in runtimes:
            runtime.close()


def close_moonep_runtimes() -> None:
    """Release all MoonEP VMM mappings before process-group destruction."""

    MoonEPRuntimeRegistry.close_all()


class _MoonEPDispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        route_weights,
        dispatched_hidden,
        dispatched_route_weights,
        runtime,
        plan,
    ):
        ctx.runtime = runtime
        ctx.plan = plan
        return dispatched_hidden, dispatched_route_weights

    @staticmethod
    def backward(ctx, grad_hidden, grad_route_weights):
        if grad_hidden is None:
            raise RuntimeError("MoonEP dispatch backward requires hidden-state gradients")
        grad_input, grad_probs = ctx.runtime.combine(
            ctx.plan, grad_hidden, grad_route_weights
        )
        return grad_input, grad_probs, None, None, None, None


class _MoonEPCombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, combined_output, runtime, plan):
        ctx.runtime = runtime
        ctx.plan = plan
        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.runtime.redispatch(ctx.plan, grad_output), None, None, None


class _MoonEPWeightGradBarrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, backend, runtime, plan, *home_parameters):
        ctx.backend = backend
        ctx.runtime = runtime
        ctx.plan = plan
        ctx.home_parameters = home_parameters
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        ctx.backend.fold_weight_grads(ctx.runtime, ctx.plan)
        # DDP's AccumulateGrad hook still needs to fire for every checkpoint
        # parameter. The real FP32 values are already in main_grad.
        ready_grads = tuple(torch.zeros_like(param) for param in ctx.home_parameters)
        return grad_hidden, None, None, None, *ready_grads


class MoonEPGroupedMLP:
    """Parameterless TE execution graph backed by a MoonEP VMM weight shadow."""

    def __init__(
        self,
        experts: TEGroupedMLP,
        config: TransformerConfig,
        group: ProcessGroup,
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> None:
        if not isinstance(experts, TEGroupedMLP):
            raise TypeError("MoonEP requires TEGroupedMLP experts")
        self.experts = experts
        self.config = config
        self.group = group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // ep_size
        self.num_spare_experts = self.experts_per_rank
        self.table = None
        self.ops = None
        self.runtime_weights: list[torch.Tensor] = []
        self._active_plan = None

    def _local_projection_weights(self) -> list[list[torch.Tensor]]:
        indices = list(range(self.experts_per_rank))
        return [
            self.experts.get_expert_weights("fc1", indices),
            self.experts.get_expert_weights("fc2", indices),
        ]

    @property
    def home_parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            weight
            for projection in self._local_projection_weights()
            for weight in projection
        )

    def _run_param_gather_hooks(self) -> None:
        for expert_layer in (self.experts.linear_fc1, self.experts.linear_fc2):
            for module in expert_layer.modules():
                for hook_id, hook in module._forward_pre_hooks.items():
                    if hook_id in module._forward_pre_hooks_with_kwargs:
                        result = hook(module, (), {})
                    else:
                        result = hook(module, ())
                    if result is not None:
                        raise RuntimeError(
                            "MoonEP does not support parameter pre-forward hooks that modify inputs"
                        )

    def _ensure_table(self) -> None:
        if self.table is not None:
            return
        moonep = _import_moonep()
        local_weights = self._local_projection_weights()
        for weight in (item for projection in local_weights for item in projection):
            if getattr(weight, "__fsdp_param__", False):
                raise RuntimeError("MoonEP does not support MCore FSDP")
            if not weight.is_cuda or weight.dtype != torch.bfloat16:
                raise ValueError("MoonEP expert parameters must be CUDA BF16 tensors")
        projection_shapes = [weights[0].shape for weights in local_weights]
        self.table = moonep.DistributedExpertTable(
            self.num_experts,
            self.num_spare_experts,
            projection_shapes,
            rank=self.ep_rank,
            world_size=self.ep_size,
            group=self.group,
        )
        MoonEPRuntimeRegistry.register_resource(self)
        self.ops = self._build_ops()

    @staticmethod
    def _attach_runtime_weight(op, name: str, weight: torch.Tensor, main_grad) -> None:
        if name in op._parameters:
            delattr(op, name)
        # TE's OperationFuser snapshots registered parameters when it is first
        # called. This execution graph is deliberately not attached as a child
        # module of BalancedMoELayer, so registering these VMM views here gives
        # TE proper autograd inputs without exposing them to state_dict/optimizer.
        runtime_weight = torch.nn.Parameter(weight.detach(), requires_grad=True)
        runtime_weight.main_grad = main_grad
        runtime_weight.grad_added_to_main_grad = False
        runtime_weight.zero_out_wgrad = True
        setattr(op, name, runtime_weight)

    def _build_ops(self):
        if not is_te_min_version("2.14.0"):
            raise RuntimeError("MoonEPGroupedMLP requires Transformer Engine >= 2.14")
        try:
            import transformer_engine as te
        except ImportError as exc:
            raise RuntimeError("MoonEPGroupedMLP requires Transformer Engine") from exc

        if self.config.activation_func != F.silu or not self.config.gated_linear_unit:
            raise ValueError("MoonEPGroupedMLP first version requires SwiGLU")

        num_groups = self.num_experts + self.num_spare_experts
        fc1_shape = self.table.full_weights[0].shape[1:]
        fc2_shape = self.table.full_weights[1].shape[1:]
        ops = te.pytorch.ops.Sequential()
        fc1 = te.pytorch.ops.GroupedLinear(
            num_groups,
            in_features=fc1_shape[1],
            out_features=fc1_shape[0],
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
            accumulate_into_main_grad=True,
            single_grouped_weight=False,
        )
        for index in range(num_groups):
            weight = self.table.full_weights[0][index]
            main_grad = self.table.full_grads[0][index]
            self._attach_runtime_weight(fc1, f"weight{index}", weight, main_grad)
            self.runtime_weights.append(getattr(fc1, f"weight{index}"))
        ops.append(fc1)

        activation_kwargs = {}
        if "glu_interleave_size" in inspect.signature(
            te.pytorch.ops.ScaledSwiGLU
        ).parameters:
            activation_kwargs["glu_interleave_size"] = self.config.moe_mlp_glu_interleave_size
        ops.append(te.pytorch.ops.ScaledSwiGLU(**activation_kwargs))

        fc2 = te.pytorch.ops.GroupedLinear(
            num_groups,
            in_features=fc2_shape[1],
            out_features=fc2_shape[0],
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
            accumulate_into_main_grad=True,
            single_grouped_weight=False,
        )
        for index in range(num_groups):
            weight = self.table.full_weights[1][index]
            main_grad = self.table.full_grads[1][index]
            self._attach_runtime_weight(fc2, f"weight{index}", weight, main_grad)
            self.runtime_weights.append(getattr(fc2, f"weight{index}"))
        ops.append(fc2)
        return ops

    def refresh_shadow(self) -> None:
        self._run_param_gather_hooks()
        self._ensure_table()
        self.table.refresh_home_weights(self._local_projection_weights())

    def begin(self, plan) -> None:
        if self._active_plan is not None:
            raise RuntimeError(
                "MoonEP does not support a second layer forward before the previous backward"
            )
        for runtime_weight in self.runtime_weights:
            runtime_weight.grad = None
            runtime_weight.grad_added_to_main_grad = False
        self._active_plan = plan

    def forward(
        self,
        dispatched_hidden: torch.Tensor,
        dispatched_probs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        runtime: MoonEPRuntime,
        plan,
    ) -> torch.Tensor:
        self.begin(plan)
        zero = torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        tokens_per_group = torch.diff(torch.cat((zero, cu_seqlens))).contiguous()
        tail_rows = dispatched_hidden.shape[0] - cu_seqlens[-1]
        tokens_per_group[-1].add_(tail_rows)
        valid_rows = (
            torch.arange(dispatched_hidden.shape[0], device=dispatched_hidden.device)
            < cu_seqlens[-1]
        )
        guarded_hidden = _MoonEPWeightGradBarrier.apply(
            dispatched_hidden, self, runtime, plan, *self.home_parameters
        )
        guarded_hidden = guarded_hidden * valid_rows.unsqueeze(-1)
        dispatched_probs = dispatched_probs * valid_rows
        return self.ops(
            guarded_hidden,
            tokens_per_group,
            dispatched_probs,
            tokens_per_group,
        )

    def fold_weight_grads(self, runtime: MoonEPRuntime, plan) -> None:
        if plan is not self._active_plan:
            raise RuntimeError("MoonEP weight-gradient plan lease is stale")
        runtime.reduce_grads(plan, self.table)
        local_slice = self.table.local_home_slice
        local_projection_weights = self._local_projection_weights()
        for full_grad, local_weights in zip(
            self.table.full_grads, local_projection_weights
        ):
            for shadow_grad, home_weight in zip(full_grad[local_slice], local_weights):
                if (
                    not hasattr(home_weight, "main_grad")
                    or home_weight.main_grad is None
                    or home_weight.main_grad.dtype != torch.float32
                ):
                    home_weight.main_grad = torch.zeros_like(home_weight, dtype=torch.float32)
                home_weight.main_grad.add_(shadow_grad)
                home_weight.grad_added_to_main_grad = True
        self.table.clear_local_home_grads()
        for runtime_weight in self.runtime_weights:
            runtime_weight.grad_added_to_main_grad = False
        self._active_plan = None

    def close(self) -> None:
        self.ops = None
        self.runtime_weights.clear()
        if self.table is not None:
            self.table.close()
            self.table = None
        self._active_plan = None


class MoonEPBalancedDataPlane:
    """Layer-owned orchestration for MoonEP communication and TE expert execution."""

    def __init__(
        self,
        experts: TEGroupedMLP,
        config: TransformerConfig,
        group: ProcessGroup,
        ep_rank: int,
        ep_size: int,
        num_experts: int,
    ) -> None:
        self.config = config
        self.group = group
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_spare_experts = num_experts // ep_size
        self.grouped_mlp = MoonEPGroupedMLP(
            experts, config, group, ep_rank, ep_size, num_experts
        )

    def forward(
        self,
        hidden: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        *,
        refresh_shadow: bool,
    ) -> tuple[torch.Tensor, object]:
        hidden_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden_shape[-1]).contiguous()
        route_weights, expert_ids, tokens_per_expert = adapt_mcore_route(
            probs, routing_map, self.config.moe_router_topk
        )
        runtime = MoonEPRuntimeRegistry.get(
            config=self.config,
            group=self.group,
            ep_size=self.ep_size,
            num_tokens=hidden_2d.shape[0],
            hidden_size=hidden_2d.shape[1],
            topk=self.config.moe_router_topk,
            num_experts=self.num_experts,
            num_spare_experts=self.num_spare_experts,
        )
        if refresh_shadow or self.grouped_mlp.table is None:
            self.grouped_mlp.refresh_shadow()

        dispatched_hidden, dispatched_probs, cu_seqlens, plan = (
            runtime.dispatch_and_prefetch(
                hidden_2d,
                route_weights,
                expert_ids,
                tokens_per_expert,
                self.grouped_mlp.table.full_weights,
            )
        )
        dispatched_hidden, dispatched_probs = _MoonEPDispatchAutograd.apply(
            hidden_2d,
            route_weights,
            dispatched_hidden,
            dispatched_probs,
            runtime,
            plan,
        )
        expert_output = self.grouped_mlp.forward(
            dispatched_hidden, dispatched_probs, cu_seqlens, runtime, plan
        )
        combined, _ = runtime.combine(plan, expert_output)
        output = _MoonEPCombineAutograd.apply(
            expert_output, combined, runtime, plan
        )
        return output.view(hidden_shape), plan
