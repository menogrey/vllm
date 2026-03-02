# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OOT (Out-Of-Tree) Plugin System for vLLM Linear Kernels.

This module provides a plugin registration system that allows OOT hardware vendors
to register their own kernel implementations without modifying the core vLLM code.

The OOT kernels will be automatically considered after built-in kernels during
kernel selection.

Usage:
    1. Create a kernel class that implements the appropriate kernel interface
    2. Call register_linear_kernel() to register the plugin
    3. Import this module before using vLLM

Example:
    from vllm.model_executor.kernels.linear.mixed_precision import MPLinearKernel
    from vllm.model_executor.kernels.linear.oot_plugin import register_linear_kernel
    from vllm.platforms import PlatformEnum
    
    class MyCustomKernel(MPLinearKernel):
        @classmethod
        def get_min_capability(cls) -> int:
            return 80
        
        @classmethod
        def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
            return True, None
        
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            pass
        
        def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                         bias: torch.Tensor | None = None) -> torch.Tensor:
            pass
    
    register_linear_kernel(
        kernel_class=MyCustomKernel,
        platform=PlatformEnum.CUDA,
    )
"""

import logging
from typing import TYPE_CHECKING

from vllm.platforms import PlatformEnum

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        ScaledMMLinearKernel,
    )

logger = logging.getLogger(__name__)

# Global plugin registry
# OOT kernels are added after built-in kernels during selection
_OOT_MP_KERNELS: dict[PlatformEnum, list[type]] = {}
_OOT_FP8_KERNELS: dict[PlatformEnum, list[type]] = {}
_OOT_INT8_KERNELS: dict[PlatformEnum, list[type]] = {}


def register_linear_kernel(
    kernel_class: type,
    platform: PlatformEnum,
    kernel_type: str = "mp",
) -> None:
    """
    Register a custom linear kernel for a specific platform.
    
    OOT kernels are added to the end of the kernel candidate list,
    meaning they will be tried after all built-in kernels.
    
    Args:
        kernel_class: The kernel class to register (must implement the 
                     appropriate kernel interface)
        platform: The platform enum (CUDA, ROCM, CPU, XPU, etc.)
        kernel_type: Type of kernel ("mp", "fp8", or "int8")
    
    Example:
        from vllm.platforms import PlatformEnum
        from vllm.model_executor.kernels.linear.oot_plugin import register_linear_kernel
        
        register_linear_kernel(
            kernel_class=MyCustomKernel,
            platform=PlatformEnum.CUDA,
            kernel_type="mp"
        )
    """
    if kernel_type == "mp":
        if platform not in _OOT_MP_KERNELS:
            _OOT_MP_KERNELS[platform] = []
        _OOT_MP_KERNELS[platform].append(kernel_class)
    elif kernel_type == "fp8":
        if platform not in _OOT_FP8_KERNELS:
            _OOT_FP8_KERNELS[platform].append(kernel_class)
    elif kernel_type == "int8":
        if platform not in _OOT_INT8_KERNELS:
            _OOT_INT8_KERNELS[platform].append(kernel_class)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                        f"Valid types: mp, fp8, int8")
    
    logger.info(
        "Registered OOT kernel %s for platform %s (type: %s)",
        kernel_class.__name__,
        platform.name,
        kernel_type,
    )


def get_oot_mp_kernels(platform: PlatformEnum) -> list[type]:
    """Get registered OOT MP kernels for a platform."""
    return _OOT_MP_KERNELS.get(platform, [])


def get_oot_fp8_kernels(platform: PlatformEnum) -> list[type]:
    """Get registered OOT FP8 kernels for a platform."""
    return _OOT_FP8_KERNELS.get(platform, [])


def get_oot_int8_kernels(platform: PlatformEnum) -> list[type]:
    """Get registered OOT Int8 kernels for a platform."""
    return _OOT_INT8_KERNELS.get(platform, [])


def clear_oot_kernels() -> None:
    """Clear all registered OOT kernels. Mainly for testing purposes."""
    global _OOT_MP_KERNELS, _OOT_FP8_KERNELS, _OOT_INT8_KERNELS
    _OOT_MP_KERNELS = {}
    _OOT_FP8_KERNELS = {}
    _OOT_INT8_KERNELS = {}
    logger.debug("Cleared all OOT kernels")


__all__ = [
    "register_linear_kernel",
    "get_oot_mp_kernels",
    "get_oot_fp8_kernels",
    "get_oot_int8_kernels",
    "clear_oot_kernels",
]