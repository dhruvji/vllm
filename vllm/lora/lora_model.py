# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import (
    get_lora_id,
    is_base_embedding_weights,
    parse_fine_tuned_lora_name,
)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.path import Path
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.adapter_manager import _read_file_bytes, _resolve_adapter_file
from safetensors.torch import load as safetensors_load

logger = init_logger(__name__)


class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
        self,
        lora_model_id: int,
        rank: int,
        loras: dict[str, LoRALayerWeights],
    ) -> None:
        """
        Args:
            lora_model_id: The integer id for the lora model.
            rank: lora rank.
            loras: module name -> weights for lora-replaced layers.

        """
        self.id = lora_model_id

        assert lora_model_id > 0, (
            f"a valid lora id should be greater than 0, got {self.id}"
        )
        self.rank = rank
        self.loras: dict[str, LoRALayerWeights] = loras

    def clone(self, lora_model_id: int) -> "LoRAModel":
        """Return a copy of the object with different ids.

        Will share the underlying tensors."""
        return self.__class__(
            lora_model_id,
            rank=self.rank,
            loras=self.loras.copy(),
        )

    def get_lora(self, module_name: str) -> LoRALayerWeights | None:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    def check_lora_name(self, lora_name: str) -> bool:
        return lora_name in self.loras

    @staticmethod
    def _should_skip_module(module_name: str, skip_prefixes: list[str]) -> bool:
        """Check if a module should be skipped based on skip prefixes"""
        for prefix in skip_prefixes:
            if f".{prefix}" in module_name or module_name.startswith(prefix):
                return True
        return False

    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        skip_prefixes: list[str] | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        loras: dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            if is_base_embedding_weights(tensor_name):
                continue
            # Skip modules based on model-defined prefixes (e.g., MTP layers)
            if skip_prefixes and cls._should_skip_module(tensor_name, skip_prefixes):
                continue
            module_name, is_lora_a = parse_fine_tuned_lora_name(
                tensor_name, weights_mapper
            )
            if module_name not in loras:
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper
                )

            if is_lora_a:
                if (
                    "lora_embedding_A" in tensor_name
                    and model_vocab_size is not None
                    and model_vocab_size != tensor.shape[1]
                ):
                    raise RuntimeError(
                        f"The embedding LoRA size({tensor.shape[1]}) must be consistent"
                        f" with the base model's vocabulary size({model_vocab_size})."
                    )
                loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
                if pin_memory:
                    loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)

                if pin_memory:
                    loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()

        return cls(lora_model_id, peft_helper.r, loras)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        skip_prefixes: list[str] | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.
            skip_prefixes: List of module name prefixes to skip during loading.
                Models can define this to skip modules not used in inference
                (e.g., MTP layers). Format: ["mtp."]

        Returns:
            Loaded LoRA Model.
        """
        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            module_sides: dict[str, set[str]] = {}
            for lora_module in modules.keys():  # noqa
                if is_base_embedding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Skip modules based on model-defined prefixes
                if skip_prefixes and cls._should_skip_module(
                    lora_module, skip_prefixes
                ):
                    continue
                module_name, is_lora_a = parse_fine_tuned_lora_name(
                    lora_module, weights_mapper
                )
                module_sides.setdefault(module_name, set()).add(
                    "A" if is_lora_a else "B"
                )
                # Case for expert lora weights
                if ".experts" in module_name:
                    expert_idx = module_name.find(".experts")
                    expert_suffix = module_name[expert_idx + 1 :]
                    if expert_suffix not in expected_lora_modules:
                        unexpected_modules.append(module_name)

                elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
                        unexpected_modules.append(module_name)

            incomplete_modules = sorted(
                module_name
                for module_name, sides in module_sides.items()
                if sides != {"A", "B"}
            )
            if incomplete_modules:
                missing = {
                    module_name: sorted({"A", "B"} - module_sides[module_name])
                    for module_name in incomplete_modules
                }
                raise ValueError(
                    f"While loading {lora_dir}, found incomplete LoRA modules "
                    f"(missing A/B tensor pair): {missing}."
                )

            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            lora_tensor_path = os.path.join(
                tensorizer_config.tensorizer_dir, "adapter_model.tensors"
            )
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                **tensorizer_args.deserialization_kwargs,
            )
            check_unexpected_modules(tensors)

        else:
            # Resolve the best adapter file in the directory
            resolved_path = _resolve_adapter_file(lora_dir)

            if resolved_path.endswith(".safetensors"):
                # Avoid mmap on Modal volumes and hot-checkpoint paths:
                # read bytes first, then load from buffer.
                tensor_bytes = _read_file_bytes(resolved_path)
                tensors = safetensors_load(tensor_bytes)
            elif resolved_path.endswith((".bin", ".pt")):
                with Path(resolved_path).open("rb") as f:
                    tensors = torch.load(
                        f, map_location=device, weights_only=True
                    )
            else:
                raise ValueError(f"{lora_dir} doesn't contain tensors")
            check_unexpected_modules(tensors)

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )
