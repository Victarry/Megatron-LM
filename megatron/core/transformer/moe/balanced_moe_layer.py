# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Balanced MoE layer built from existing MCore MoE components."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch

from megatron.core import tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.expert_weight_dispatcher import (
    AllToAllExpertWeightDispatcher,
    HybridEPExpertWeightDispatcher,
    SymmetricMemoryExpertWeightDispatcher,
    _runtime_weight_grad_edge_dtype,
)
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.offloading_planner import (
    gen_offloading_plan,
    gen_random_offloading_plan,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none


@dataclass(frozen=True)
class BalancedMoEDebugStats:
    """Scalar debug stats captured from the most recent forward pass."""

    num_local_active_spare_slots: int
    num_global_active_spare_slots: int
    num_moved_token_assignments: int
    max_rank_load_before: int
    max_rank_load_after: int


@dataclass(frozen=True)
class _RuntimeMainGradFoldbackContext:
    module: str
    metadata: object
    home_weights: list[torch.Tensor]
    spare_weights: list[torch.Tensor]


class _RuntimeExpertWeightForwardCast(torch.autograd.Function):
    """Use low-precision weights in expert GEMMs while reducing their gradients in FP32."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, forward_dtype: torch.dtype) -> torch.Tensor:
        ctx.backward_dtype = weight.dtype
        return weight.to(dtype=forward_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.to(dtype=ctx.backward_dtype), None


def _cast_runtime_expert_weights_for_forward(
    expert_weights: list[torch.Tensor], forward_dtype: torch.dtype
) -> list[torch.Tensor]:
    """Cast runtime weights for expert forward while preserving an FP32 grad edge."""

    return [
        (
            _RuntimeExpertWeightForwardCast.apply(expert_weight, forward_dtype)
            if expert_weight.dtype != forward_dtype
            else expert_weight
        )
        for expert_weight in expert_weights
    ]


class BalancedMoELayer(BaseMoELayer):
    """MoE layer that executes selected routed assignments on runtime spare slots."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        self.submodules = not_none(submodules)
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        self.pg_collection = pg_collection
        self._validate_config(config)

        super().__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.ep_size = utils.get_pg_size(self.ep_group)
        self.ep_rank = utils.get_pg_rank(self.ep_group)
        self.balance_backend = getattr(self.config, "moe_balance_backend", "legacy")
        self.num_home_experts = not_none(self.config.num_moe_experts)
        if self.balance_backend == "moonep":
            self._init_moonep(name, pg_collection)
            return

        self.num_spare_experts = not_none(getattr(self.config, "moe_num_spare_experts", None))
        self.num_local_home_experts = self.num_home_experts // self.ep_size
        self.num_local_spare_experts = self.num_spare_experts // self.ep_size
        self.num_local_total_experts = self.num_local_home_experts + self.num_local_spare_experts
        self.num_total_experts = self.num_home_experts + self.num_spare_experts
        self.local_home_expert_indices = list(range(self.num_local_home_experts))
        self.local_spare_expert_indices = list(
            range(self.num_local_home_experts, self.num_local_total_experts)
        )
        self.local_expert_indices = [
            self.ep_rank * self.num_local_total_experts + i
            for i in range(self.num_local_total_experts)
        ]
        self.num_local_experts = self.num_local_total_experts
        self.last_debug_stats: BalancedMoEDebugStats | None = None
        self._last_runtime_spare_main_grad_dtypes: list[torch.dtype] = []
        self._last_runtime_spare_main_grad_abs_sums: list[float] = []
        self._last_runtime_folded_home_grad_dtypes: list[torch.dtype] = []

        self.router = self.submodules.router(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
        )

        dispatcher_config = copy(self.config)
        dispatcher_config.num_moe_experts = self.num_total_experts
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            self.num_local_total_experts,
            self.local_expert_indices,
            config=dispatcher_config,
            pg_collection=pg_collection,
        )

        self.experts = self.submodules.experts(
            self.num_local_total_experts,
            self.config,
            pg_collection=pg_collection,
            name=(name + ".experts") if name is not None else None,
        )
        self._runtime_weight_main_grad_accumulation = False
        if hasattr(self.experts, "enable_runtime_weight_main_grad_accumulation"):
            self.experts.enable_runtime_weight_main_grad_accumulation()
            self._runtime_weight_main_grad_accumulation = True
        elif hasattr(self.experts, "disable_runtime_weight_main_grad_accumulation"):
            self.experts.disable_runtime_weight_main_grad_accumulation()
        self.experts.free_expert_parameters(self.local_spare_expert_indices)

        self.expert_weight_dispatcher = self._make_expert_weight_dispatcher()

        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            shared_expert_kwargs = {
                "config": self.config,
                "pg_collection": pg_collection,
                "gate": self.config.moe_shared_expert_gate,
            }
            if self._builder_accepts_name(self.submodules.shared_experts):
                shared_expert_kwargs["name"] = (
                    name + ".shared_experts" if name is not None else None
                )
            self.shared_experts = self.submodules.shared_experts(**shared_expert_kwargs)

    def _init_moonep(
        self,
        name: str | None,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        """Initialize checkpoint-owned experts and the MoonEP execution data plane."""

        from megatron.core.transformer.moe.moonep_backend import MoonEPBalancedDataPlane

        self.num_spare_experts = 0
        self.num_local_home_experts = self.num_home_experts // self.ep_size
        self.num_local_spare_experts = 0
        self.num_local_total_experts = self.num_local_home_experts
        self.num_total_experts = self.num_home_experts
        self.local_home_expert_indices = list(range(self.num_local_home_experts))
        self.local_spare_expert_indices = []
        self.local_expert_indices = [
            self.ep_rank * self.num_local_home_experts + i
            for i in range(self.num_local_home_experts)
        ]
        self.num_local_experts = self.num_local_home_experts
        self.last_debug_stats = None
        self._last_runtime_spare_main_grad_dtypes = []
        self._last_runtime_spare_main_grad_abs_sums = []
        self._last_runtime_folded_home_grad_dtypes = []
        self.is_first_microbatch = True

        self.router = self.submodules.router(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=self.is_mtp_layer,
            layer_number=self.layer_number,
        )
        self.experts = self.submodules.experts(
            self.num_local_home_experts,
            self.config,
            pg_collection=pg_collection,
            name=(name + ".experts") if name is not None else None,
        )
        self.token_dispatcher = None
        self.expert_weight_dispatcher = None
        self.moonep_data_plane = MoonEPBalancedDataPlane(
            self.experts,
            self.config,
            self.ep_group,
            self.ep_rank,
            self.ep_size,
            self.num_home_experts,
        )

        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            shared_expert_kwargs = {
                "config": self.config,
                "pg_collection": pg_collection,
                "gate": self.config.moe_shared_expert_gate,
            }
            if self._builder_accepts_name(self.submodules.shared_experts):
                shared_expert_kwargs["name"] = (
                    name + ".shared_experts" if name is not None else None
                )
            self.shared_experts = self.submodules.shared_experts(**shared_expert_kwargs)

    @staticmethod
    def _builder_accepts_name(builder) -> bool:
        try:
            return "name" in inspect.signature(builder).parameters
        except (TypeError, ValueError):
            return True

    def _make_expert_weight_dispatcher(self):
        backend = getattr(self.config, "moe_balance_expert_weight_dispatch_backend", "all_to_all")
        dispatcher_kwargs = {
            "config": self.config,
            "ep_group": self.ep_group,
            "num_home_experts": self.num_home_experts,
            "num_spare_experts": self.num_spare_experts,
        }
        if backend == "all_to_all":
            return AllToAllExpertWeightDispatcher(**dispatcher_kwargs)
        if backend == "symmetric_memory":
            availability_error = SymmetricMemoryExpertWeightDispatcher.availability_error()
            if availability_error is not None:
                raise RuntimeError(
                    "BalancedMoELayer symmetric_memory expert-weight dispatch is unavailable: "
                    f"{availability_error}"
                )
            return SymmetricMemoryExpertWeightDispatcher(**dispatcher_kwargs)
        if backend == "hybridep":
            availability_error = HybridEPExpertWeightDispatcher.availability_error()
            if availability_error is not None:
                raise RuntimeError(
                    "BalancedMoELayer hybridep expert-weight dispatch is unavailable: "
                    f"{availability_error}"
                )
            return HybridEPExpertWeightDispatcher(**dispatcher_kwargs)
        raise ValueError(
            "BalancedMoELayer requires moe_balance_expert_weight_dispatch_backend to be "
            "'all_to_all', 'symmetric_memory', or 'hybridep'."
        )

    @staticmethod
    def _validate_config(config: TransformerConfig) -> None:
        if not getattr(config, "moe_use_balanced_layer", False):
            return
        balance_backend = getattr(config, "moe_balance_backend", "legacy")
        if balance_backend == "moonep":
            if config.num_moe_experts is None:
                raise ValueError("MoonEP BalancedMoELayer requires num_moe_experts.")
            if config.expert_model_parallel_size <= 1:
                raise ValueError("MoonEP BalancedMoELayer requires EP > 1.")
            if config.num_moe_experts % config.expert_model_parallel_size != 0:
                raise ValueError("MoonEP requires num_moe_experts divisible by EP.")
            if not config.bf16 or config.fp16:
                raise ValueError("MoonEP requires BF16 precision.")
            if config.moe_router_dtype != "fp32":
                raise ValueError("MoonEP requires moe_router_dtype='fp32'.")
            if not config.moe_grouped_gemm:
                raise ValueError("MoonEP requires TEGroupedMLP.")
            if config.add_bias_linear:
                raise ValueError("MoonEP does not support expert bias.")
            if getattr(config, "moe_num_spare_experts", None) is not None:
                raise ValueError("moe_num_spare_experts is legacy-only under MoonEP.")
            if getattr(config, "moe_balance_recompute_expert_dispatch", False):
                raise ValueError("MoonEP does not support expert-dispatch recompute.")
            if config.moe_expert_capacity_factor is not None or config.moe_token_dropping:
                raise ValueError("MoonEP does not support token dropping.")
            if config.moe_shared_expert_overlap:
                raise ValueError("MoonEP only supports non-overlap shared experts.")
            if config.cuda_graph_impl != "none":
                raise ValueError("MoonEP does not support CUDA Graph.")
            if config.fp8 is not None or config.fp4 is not None:
                raise ValueError("MoonEP does not support FP8 or FP4.")
            if config.expert_tensor_parallel_size != 1:
                raise ValueError("MoonEP does not support expert tensor parallelism.")
            if config.pipeline_model_parallel_size != 1:
                raise ValueError("MoonEP first version requires PP1.")
            if config.tensor_model_parallel_size != 1:
                raise ValueError("MoonEP first version requires TP1.")
            return
        if balance_backend != "legacy":
            raise ValueError("moe_balance_backend must be 'legacy' or 'moonep'.")

        expert_weight_backend = getattr(
            config, "moe_balance_expert_weight_dispatch_backend", "all_to_all"
        )
        grad_combine_dtype = getattr(
            config, "moe_balance_expert_weight_grad_combine_dtype", "fp32"
        )
        if expert_weight_backend not in (
            "all_to_all",
            "symmetric_memory",
            "hybridep",
        ):
            raise ValueError(
                "BalancedMoELayer requires moe_balance_expert_weight_dispatch_backend "
                "to be 'all_to_all', 'symmetric_memory', or 'hybridep'."
            )
        if grad_combine_dtype not in (
            "fp32",
            "param_dtype",
            "bf16",
            "fp16",
        ):
            raise ValueError(
                "BalancedMoELayer requires moe_balance_expert_weight_grad_combine_dtype "
                "to be 'fp32', 'param_dtype', 'bf16', or 'fp16'."
            )
        if expert_weight_backend == "hybridep":
            if grad_combine_dtype != "param_dtype":
                raise ValueError(
                    "BalancedMoELayer HybridEP expert-weight dispatch currently requires "
                    "moe_balance_expert_weight_grad_combine_dtype='param_dtype' because "
                    "HybridEP combines gradients in the parameter dtype."
                )
            if config.moe_grouped_gemm:
                raise ValueError(
                    "BalancedMoELayer HybridEP expert-weight dispatch does not yet support "
                    "TEGroupedMLP runtime main_grad foldback."
                )
        if config.moe_grouped_gemm and getattr(
            config, "moe_balance_recompute_expert_dispatch", False
        ):
            raise ValueError(
                "BalancedMoELayer does not support recompute_expert_dispatch with "
                "TEGroupedMLP runtime main_grad foldback."
            )
        if config.num_moe_experts is None:
            raise ValueError("BalancedMoELayer requires num_moe_experts.")
        if config.moe_ffn_hidden_size is None:
            raise ValueError("BalancedMoELayer requires moe_ffn_hidden_size.")
        if config.expert_model_parallel_size <= 1:
            raise ValueError("BalancedMoELayer requires expert_model_parallel_size > 1.")
        if config.num_moe_experts % config.expert_model_parallel_size != 0:
            raise ValueError("BalancedMoELayer requires num_moe_experts divisible by EP size.")
        num_spare_experts = getattr(config, "moe_num_spare_experts", None)
        if num_spare_experts is None or num_spare_experts <= 0:
            raise ValueError("BalancedMoELayer requires a positive moe_num_spare_experts.")
        if num_spare_experts % config.expert_model_parallel_size != 0:
            raise ValueError(
                "BalancedMoELayer requires moe_num_spare_experts divisible by EP size."
            )
        if config.moe_token_dispatcher_type != "alltoall":
            raise ValueError("BalancedMoELayer requires moe_token_dispatcher_type='alltoall'.")
        spare_per_rank = num_spare_experts // config.expert_model_parallel_size
        if (
            getattr(config, "moe_balance_assignment_algorithm", "approx_bin_packing")
            == "approx_bin_packing"
            and spare_per_rank != 1
        ):
            raise ValueError("BalancedMoELayer approx_bin_packing supports one spare per EP rank.")
        if config.cuda_graph_impl != "none":
            raise ValueError("BalancedMoELayer does not support CUDA Graph in the MVP.")
        if config.fp8 is not None or config.fp4 is not None:
            raise ValueError("BalancedMoELayer does not support FP8 or FP4 in the MVP.")
        if config.moe_router_padding_for_fp8 or config.moe_router_padding_for_quantization:
            raise ValueError("BalancedMoELayer does not support quantization router padding.")
        if config.transformer_impl == "inference_optimized":
            raise ValueError("BalancedMoELayer does not support inference_optimized.")
        if config.expert_tensor_parallel_size != 1:
            raise ValueError("BalancedMoELayer requires expert_tensor_parallel_size=1.")
        if config.moe_latent_size is not None:
            raise ValueError("BalancedMoELayer does not support moe_latent_size.")
        if config.moe_paged_stash:
            raise ValueError("BalancedMoELayer does not support moe_paged_stash.")
        if config.moe_shared_expert_overlap:
            raise ValueError("BalancedMoELayer does not support shared expert overlap.")
        if config.overlap_moe_expert_parallel_comm:
            raise ValueError("BalancedMoELayer does not support MoE EP communication overlap.")
        if config.delay_wgrad_compute:
            raise ValueError("BalancedMoELayer does not support delayed wgrad compute.")
        if config.overlap_dispatch_backward_with_experts_wgrad:
            raise ValueError("BalancedMoELayer does not support dispatch-backward overlap.")

    def _gather_tokens_per_home_expert(self, routing_map: torch.Tensor) -> torch.Tensor:
        local_counts = routing_map.sum(dim=0).to(torch.int32)
        if self.ep_size == 1:
            return local_counts.unsqueeze(0)
        gathered_counts = [torch.empty_like(local_counts) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_counts, local_counts, group=self.ep_group)
        return torch.stack(gathered_counts, dim=0)

    def _make_plan(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        tokens_per_expert_from_ep_rank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self.config, "moe_balance_enable_random_offloading", False):
            return gen_random_offloading_plan(
                routing_map,
                probs,
                tokens_per_expert_from_ep_rank,
                ep_rank=self.ep_rank,
                ep=self.ep_size,
                spare_expert_per_ep_rank=self.num_local_spare_experts,
                threshold_multiplier=getattr(self.config, "moe_balance_threshold_multiplier", 0.0),
            )
        return gen_offloading_plan(
            routing_map,
            probs,
            tokens_per_expert_from_ep_rank,
            ep_rank=self.ep_rank,
            num_ep_ranks=self.ep_size,
            num_spare_experts_per_ep_rank=self.num_local_spare_experts,
            threshold_multiplier=getattr(self.config, "moe_balance_threshold_multiplier", 0.0),
            assignment_algorithm=getattr(
                self.config, "moe_balance_assignment_algorithm", "approx_bin_packing"
            ),
        )

    def _validate_plan(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        rerouting_map: torch.Tensor,
        rerouted_probs: torch.Tensor,
        expert_offloading_map: torch.Tensor,
    ) -> None:
        expected_expert_shape = (self.num_home_experts, self.num_spare_experts)
        expected_route_shape = (routing_map.shape[0], self.num_total_experts)
        if tuple(rerouting_map.shape) != expected_route_shape:
            raise ValueError(
                f"BalancedMoELayer expected rerouting_map shape {expected_route_shape}, "
                f"got {tuple(rerouting_map.shape)}."
            )
        if tuple(rerouted_probs.shape) != expected_route_shape:
            raise ValueError(
                f"BalancedMoELayer expected rerouted_probs shape {expected_route_shape}, "
                f"got {tuple(rerouted_probs.shape)}."
            )
        if tuple(expert_offloading_map.shape) != expected_expert_shape:
            raise ValueError(
                "BalancedMoELayer expected expert_offloading_map shape "
                f"{expected_expert_shape}, got {tuple(expert_offloading_map.shape)}."
            )
        if rerouting_map.dtype != torch.bool:
            raise ValueError("BalancedMoELayer expects rerouting_map to be bool.")
        if expert_offloading_map.dtype != torch.bool:
            raise ValueError("BalancedMoELayer expects expert_offloading_map to be bool.")
        if expert_offloading_map.sum(dim=0).max().item() > 1:
            raise ValueError(
                "BalancedMoELayer expected each spare expert slot to map to at most one "
                "home expert."
            )
        if not torch.equal(rerouting_map.sum(dim=1), routing_map.sum(dim=1)):
            raise ValueError("BalancedMoELayer planner changed per-token assignment counts.")
        torch.testing.assert_close(rerouted_probs.sum(dim=1), probs.sum(dim=1), rtol=0, atol=1e-6)

    def _install_spare_expert_weights(
        self, expert_offloading_map: torch.Tensor
    ) -> tuple[list[tensor_parallel.CheckpointWithoutOutput], list[_RuntimeMainGradFoldbackContext]]:
        metadata = self.expert_weight_dispatcher.preprocess(expert_offloading_map)
        checkpoints: list[tensor_parallel.CheckpointWithoutOutput] = []
        foldback_contexts: list[_RuntimeMainGradFoldbackContext] = []
        recompute_dispatch = getattr(self.config, "moe_balance_recompute_expert_dispatch", False)
        backend = getattr(self.config, "moe_balance_expert_weight_dispatch_backend", "all_to_all")
        for module in ("fc1", "fc2"):
            home_weights = self.experts.get_expert_weights(module, self.local_home_expert_indices)
            forward_dtype = home_weights[0].dtype
            runtime_weight_dtype = (
                forward_dtype
                if self._runtime_weight_main_grad_accumulation
                else _runtime_weight_grad_edge_dtype(
                    forward_dtype,
                    getattr(self.config, "moe_balance_expert_weight_grad_combine_dtype", "fp32"),
                )
                if backend in ("all_to_all", "symmetric_memory")
                else forward_dtype
            )
            if recompute_dispatch:
                checkpoint = tensor_parallel.CheckpointWithoutOutput()
                spare_weights = list(
                    checkpoint.checkpoint(
                        partial(
                            self._dispatch_spare_expert_weights_for_checkpoint,
                            metadata,
                            runtime_weight_dtype,
                        ),
                        *home_weights,
                    )
                )
                checkpoints.append(checkpoint)
            else:
                spare_weights = self.expert_weight_dispatcher.dispatch(
                    metadata, *home_weights, runtime_weight_dtype=runtime_weight_dtype
                )
            if not self._runtime_weight_main_grad_accumulation:
                spare_weights = _cast_runtime_expert_weights_for_forward(spare_weights, forward_dtype)
            self.experts.set_expert_weights(module, spare_weights, self.local_spare_expert_indices)
            if self._runtime_weight_main_grad_accumulation:
                attached_spare_weights = self.experts.get_expert_weights(
                    module, self.local_spare_expert_indices
                )
                foldback_contexts.append(
                    _RuntimeMainGradFoldbackContext(
                        module=module,
                        metadata=metadata,
                        home_weights=home_weights,
                        spare_weights=attached_spare_weights,
                    )
                )
        return checkpoints, foldback_contexts

    def _dispatch_spare_expert_weights_for_checkpoint(
        self, metadata, runtime_weight_dtype: torch.dtype, *home_weights: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            self.expert_weight_dispatcher.dispatch(
                metadata, *home_weights, runtime_weight_dtype=runtime_weight_dtype
            )
        )

    @staticmethod
    def _ensure_main_grad(
        weight: torch.Tensor, grad: torch.Tensor, *, preferred_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        dtype = preferred_dtype or grad.dtype
        if (
            not hasattr(weight, "main_grad")
            or weight.main_grad is None
            or weight.main_grad.shape != weight.shape
            or weight.main_grad.device != weight.device
            or weight.main_grad.dtype != dtype
        ):
            weight.main_grad = torch.zeros_like(weight, dtype=dtype)
        return weight.main_grad

    def _fold_runtime_main_grads(
        self, contexts: list[_RuntimeMainGradFoldbackContext], grad_input: torch.Tensor
    ) -> torch.Tensor:
        self._last_runtime_spare_main_grad_dtypes = []
        self._last_runtime_spare_main_grad_abs_sums = []
        self._last_runtime_folded_home_grad_dtypes = []
        for context in contexts:
            spare_grads = []
            for spare_weight in context.spare_weights:
                main_grad = getattr(spare_weight, "main_grad", None)
                if main_grad is None:
                    raise RuntimeError(
                        "BalancedMoELayer expected TE runtime spare weight main_grad "
                        f"for {context.module}."
                    )
                self._last_runtime_spare_main_grad_dtypes.append(main_grad.dtype)
                self._last_runtime_spare_main_grad_abs_sums.append(
                    float(main_grad.detach().abs().sum().item())
                )
                spare_grads.append(main_grad)
            local_home_grads = self.expert_weight_dispatcher.fold_spare_gradients(
                context.metadata,
                *spare_grads,
                reference_weight_dtype=context.home_weights[0].dtype,
            )
            for home_weight, home_grad in zip(context.home_weights, local_home_grads):
                self._last_runtime_folded_home_grad_dtypes.append(home_grad.dtype)
                main_grad = self._ensure_main_grad(
                    home_weight, home_grad, preferred_dtype=torch.float32
                )
                main_grad.add_(home_grad.to(dtype=main_grad.dtype))
            for spare_weight in context.spare_weights:
                spare_weight.main_grad.zero_()
                spare_weight.grad_added_to_main_grad = False
        return grad_input

    def _effective_spare_columns(self) -> list[int]:
        columns = []
        for spare_idx in range(self.num_spare_experts):
            ep_rank = spare_idx // self.num_local_spare_experts
            local_spare_idx = spare_idx % self.num_local_spare_experts
            columns.append(
                ep_rank * self.num_local_total_experts
                + self.num_local_home_experts
                + local_spare_idx
            )
        return columns

    def _maybe_record_debug_stats(
        self,
        tokens_per_expert_from_ep_rank: torch.Tensor,
        rerouting_map: torch.Tensor,
        expert_offloading_map: torch.Tensor,
    ) -> None:
        if not getattr(self.config, "moe_balance_enable_debug_stats", False):
            self.last_debug_stats = None
            return

        local_spare_start = self.ep_rank * self.num_local_spare_experts
        local_spare_end = local_spare_start + self.num_local_spare_experts
        local_active_spare_slots = expert_offloading_map[:, local_spare_start:local_spare_end].any(
            dim=0
        )

        local_effective_counts = rerouting_map.sum(dim=0).to(torch.int32)
        if self.ep_size == 1:
            effective_counts = local_effective_counts.unsqueeze(0)
        else:
            gathered = [torch.empty_like(local_effective_counts) for _ in range(self.ep_size)]
            torch.distributed.all_gather(gathered, local_effective_counts, group=self.ep_group)
            effective_counts = torch.stack(gathered, dim=0)

        global_home_counts = tokens_per_expert_from_ep_rank.sum(dim=0)
        before_rank_load = global_home_counts.reshape(
            self.ep_size, self.num_local_home_experts
        ).sum(dim=1)
        global_effective_counts = effective_counts.sum(dim=0)
        after_rank_load = global_effective_counts.reshape(
            self.ep_size, self.num_local_total_experts
        ).sum(dim=1)
        spare_columns = self._effective_spare_columns()
        moved_assignments = int(rerouting_map[:, spare_columns].sum().item())

        self.last_debug_stats = BalancedMoEDebugStats(
            num_local_active_spare_slots=int(local_active_spare_slots.sum().item()),
            num_global_active_spare_slots=int(expert_offloading_map.any(dim=0).sum().item()),
            num_moved_token_assignments=moved_assignments,
            max_rank_load_before=int(before_rank_load.max().item()),
            max_rank_load_after=int(after_rank_load.max().item()),
        )

    def _assert_inactive_spares_receive_no_tokens(
        self, tokens_per_expert: torch.Tensor, expert_offloading_map: torch.Tensor
    ) -> None:
        if not self.local_spare_expert_indices:
            return
        local_spare_start = self.ep_rank * self.num_local_spare_experts
        local_spare_end = local_spare_start + self.num_local_spare_experts
        local_active_spare_slots = expert_offloading_map[:, local_spare_start:local_spare_end].any(
            dim=0
        )
        local_spare_counts = tokens_per_expert[self.local_spare_expert_indices].to(
            local_active_spare_slots.device
        )
        if torch.any(local_spare_counts[~local_active_spare_slots] != 0):
            raise ValueError("BalancedMoELayer routed tokens to an inactive spare expert slot.")

    def _shared_experts_compute(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.use_shared_expert and not self.shared_expert_overlap:
            return apply_module(self.shared_experts)(hidden_states)
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise ValueError("BalancedMoELayer does not support intermediate_tensors.")
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        shared_expert_output = self._shared_experts_compute(hidden_states)
        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask, input_ids)
        if self.balance_backend == "moonep":
            output, _plan = self.moonep_data_plane.forward(
                hidden_states,
                probs,
                routing_map,
                refresh_shadow=self.is_first_microbatch,
            )
            self.is_first_microbatch = False
            if shared_expert_output is not None:
                output = output + shared_expert_output
            return output, None

        tokens_per_expert_from_ep_rank = self._gather_tokens_per_home_expert(routing_map)
        rerouting_map, rerouted_probs, expert_offloading_map = self._make_plan(
            routing_map, probs, tokens_per_expert_from_ep_rank
        )
        self._validate_plan(
            routing_map, probs, rerouting_map, rerouted_probs, expert_offloading_map
        )
        self._maybe_record_debug_stats(
            tokens_per_expert_from_ep_rank, rerouting_map, expert_offloading_map
        )
        expert_dispatch_checkpoints, runtime_main_grad_contexts = self._install_spare_expert_weights(
            expert_offloading_map
        )

        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, rerouting_map, rerouted_probs
        )
        dispatched_input, probs = self.token_dispatcher.token_dispatch(hidden_states, probs)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
        )
        self._assert_inactive_spares_receive_no_tokens(tokens_per_expert, expert_offloading_map)
        if runtime_main_grad_contexts and dispatched_input.requires_grad:
            dispatched_input.register_hook(
                partial(self._fold_runtime_main_grads, runtime_main_grad_contexts)
            )
        expert_output, mlp_bias = apply_module(self.experts)(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"

        output = self.token_dispatcher.combine_preprocess(expert_output)
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)

        for checkpoint in expert_dispatch_checkpoints:
            checkpoint.discard_output_and_register_recompute(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output, None

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        """Run delayed weight-gradient computation for local expert modules."""

        if routed_experts:
            self.experts.backward_dw()
        if shared_experts and self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def _is_spare_state_key(self, key: str) -> bool:
        """Return whether a state-dict key belongs to a runtime spare expert."""

        for expert_index in self.local_spare_expert_indices:
            if f"experts.local_experts.{expert_index}." in key:
                return True
            for layer_name in ("linear_fc1", "linear_fc2"):
                for parameter_name in ("weight", "bias"):
                    if key.endswith(f"experts.{layer_name}.{parameter_name}{expert_index}"):
                        return True
        return False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return a checkpoint-compatible state dict without spare expert weights."""

        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state_dict.keys()):
            if self._is_spare_state_key(key):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load a home-expert checkpoint while materializing local spare keys."""

        if not strict:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        augmented_state_dict = OrderedDict(state_dict)
        if hasattr(state_dict, "_metadata"):
            augmented_state_dict._metadata = state_dict._metadata
        raw_state_dict = super().state_dict()
        for key, value in raw_state_dict.items():
            if self._is_spare_state_key(key) and key not in augmented_state_dict:
                augmented_state_dict[key] = value
        return super().load_state_dict(augmented_state_dict, strict=True, assign=assign)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Return a sharded state dict that is compatible with standard MoELayer keys."""

        original_values = []
        if hasattr(self.experts, "num_local_experts"):
            original_values.append(
                (self.experts, "num_local_experts", self.experts.num_local_experts)
            )
            self.experts.num_local_experts = self.num_local_home_experts
        for layer_name in ("linear_fc1", "linear_fc2"):
            expert_layer = getattr(self.experts, layer_name, None)
            if expert_layer is not None and hasattr(expert_layer, "num_gemms"):
                original_values.append((expert_layer, "num_gemms", expert_layer.num_gemms))
                expert_layer.num_gemms = self.num_local_home_experts
        try:
            sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
            for key in list(sharded_state_dict.keys()):
                if self._is_spare_state_key(key):
                    del sharded_state_dict[key]
            return sharded_state_dict
        finally:
            for module, attribute, value in original_values:
                setattr(module, attribute, value)

    def set_for_recompute_pre_mlp_layernorm(self):
        """Mark shared-expert TE layers to preserve inputs needed by recompute."""

        if self.shared_experts is not None:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
