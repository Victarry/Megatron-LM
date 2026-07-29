# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP-backed balanced mixture-of-experts layer."""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core import utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.moonep_backend import MoonEPDataPlane
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none


class BalancedMoELayer(BaseMoELayer):
    """MoE layer whose complete communication data plane is provided by MoonEP."""

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
        super().__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.router = self.submodules.router(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
        )
        self.experts = self.submodules.experts(
            self.num_local_experts,
            config,
            pg_collection=pg_collection,
            name=(name + ".experts") if name is not None else None,
        )
        self.data_plane = MoonEPDataPlane(
            experts=self.experts,
            config=config,
            group=self.ep_group,
            ep_rank=utils.get_pg_rank(self.ep_group),
            ep_size=utils.get_pg_size(self.ep_group),
            num_experts=not_none(config.num_moe_experts),
        )
        self.is_first_microbatch = True

        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is required when shared experts are configured."
            self.shared_experts = self.submodules.shared_experts(
                config=config,
                pg_collection=pg_collection,
                gate=config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> tuple[torch.Tensor, None]:
        if intermediate_tensors is not None:
            raise ValueError("BalancedMoELayer does not support partial MoE execution.")
        if packed_seq_params is not None:
            raise ValueError("BalancedMoELayer does not support packed sequences.")

        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        shared_output = (
            apply_module(self.shared_experts)(hidden_states)
            if self.shared_experts is not None
            else None
        )
        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask, input_ids)
        output = self.data_plane(
            hidden_states, probs, routing_map, refresh_shadow=self.is_first_microbatch
        )
        self.is_first_microbatch = False
        if shared_output is not None:
            output = output + shared_output
        return output, None

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False) -> None:
        """Run delayed shared-expert wgrad when requested.

        MoonEP routed-expert wgrad is produced eagerly by its private TE graph,
        and delayed routed-expert wgrad is rejected by configuration validation.
        """

        if routed_experts and self.config.delay_wgrad_compute:
            raise RuntimeError("MoonEP does not support delayed routed-expert wgrad.")
        if shared_experts and self.shared_experts is not None:
            self.shared_experts.backward_dw()
