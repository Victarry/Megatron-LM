# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Balanced MoE layer built from existing MCore MoE components."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core import utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.expert_weight_dispatcher import AllToAllExpertWeightDispatcher
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
        self.num_home_experts = not_none(self.config.num_moe_experts)
        self.num_spare_experts = not_none(getattr(self.config, "moe_num_spare_experts", None))
        self.num_local_home_experts = self.num_home_experts // self.ep_size
        self.num_local_spare_experts = self.num_spare_experts // self.ep_size
        self.num_local_total_experts = (
            self.num_local_home_experts + self.num_local_spare_experts
        )
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
        self.experts.free_expert_parameters(self.local_spare_expert_indices)

        self.expert_weight_dispatcher = AllToAllExpertWeightDispatcher(
            config=self.config,
            ep_group=self.ep_group,
            num_home_experts=self.num_home_experts,
            num_spare_experts=self.num_spare_experts,
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

    @staticmethod
    def _validate_config(config: TransformerConfig) -> None:
        if not getattr(config, "moe_use_balanced_layer", False):
            return
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
            raise ValueError("BalancedMoELayer requires moe_num_spare_experts divisible by EP size.")
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
        if not torch.equal(rerouting_map.sum(dim=1), routing_map.sum(dim=1)):
            raise ValueError("BalancedMoELayer planner changed per-token assignment counts.")
        torch.testing.assert_close(
            rerouted_probs.sum(dim=1), probs.sum(dim=1), rtol=0, atol=1e-6
        )

    def _install_spare_expert_weights(self, expert_offloading_map: torch.Tensor) -> None:
        metadata = self.expert_weight_dispatcher.preprocess(expert_offloading_map)
        for module in ("fc1", "fc2"):
            home_weights = self.experts.get_expert_weights(
                module, self.local_home_expert_indices
            )
            spare_weights = self.expert_weight_dispatcher.dispatch(metadata, *home_weights)
            self.experts.set_expert_weights(module, spare_weights, self.local_spare_expert_indices)

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
        before_rank_load = global_home_counts.reshape(self.ep_size, self.num_local_home_experts).sum(
            dim=1
        )
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
        self._install_spare_expert_weights(expert_offloading_map)

        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, rerouting_map, rerouted_probs
        )
        dispatched_input, probs = self.token_dispatcher.token_dispatch(hidden_states, probs)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(dispatched_input, probs)
        )
        self._assert_inactive_spares_receive_no_tokens(tokens_per_expert, expert_offloading_map)
        expert_output, mlp_bias = apply_module(self.experts)(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"

        output = self.token_dispatcher.combine_preprocess(expert_output)
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output, None

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        if routed_experts:
            self.experts.backward_dw()
        if shared_experts and self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def _is_spare_state_key(self, key: str) -> bool:
        for expert_index in self.local_spare_expert_indices:
            if f"experts.local_experts.{expert_index}." in key:
                return True
            for layer_name in ("linear_fc1", "linear_fc2"):
                for parameter_name in ("weight", "bias"):
                    if key.endswith(f"experts.{layer_name}.{parameter_name}{expert_index}"):
                        return True
        return False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state_dict.keys()):
            if self._is_spare_state_key(key):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
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
        original_values = []
        if hasattr(self.experts, "num_local_experts"):
            original_values.append((self.experts, "num_local_experts", self.experts.num_local_experts))
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
        if self.shared_experts is not None:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
