import functools
import logging
from typing import Callable

import safetensors.torch
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from moshi.models.lm import LMModel
from moshi.models.loaders import _is_safetensors, get_moshi_lm
from moshi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


class _TrainWrapper(torch.nn.Module):
    """Thin wrapper so FSDP hooks intercept __call__ → forward() before
    delegating to LMModel.forward_train(). Without this, calling
    model.forward_train(codes) directly bypasses FSDP's all-gather,
    exposing sharded (1-D) parameters to F.embedding."""

    def __init__(self, model: LMModel):
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor):
        return self.model.forward_train(codes)

    def __getattr__(self, name: str):
        if name == 'model' or name in self._parameters:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy() -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    Each Transformers block becomes its own FSDP group so that only a single
    Transformer block is sharded at a time.
    """

    return functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )


def log_train_params(model: torch.nn.Module | FullyShardedDataParallel):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )


def get_fsdp_model(
    args: TrainArgs, moshi_path: str | None = None
) -> FullyShardedDataParallel | LMModel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) LMModel or a non
    sharded LMModel if only one GPU is available.

    Uses meta-device initialization for memory efficiency. Parameters are
    materialised on rank 0 first, then broadcast via FSDP.
    """
    if moshi_path is None:
        moshi_path = getattr(args.moshi_paths, "moshi_path", None)

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32
    else:
        param_dtype = torch.bfloat16

    with torch.device("meta"):
        model = get_moshi_lm(None, device="meta", dtype=param_dtype)

    actual_dep_q = getattr(model, "dep_q", None)
    if actual_dep_q is not None:
        logger.info(
            f"LMModel dep_q={actual_dep_q} "
            f"(expected 16 for PersonaPlex, 8 for standard Moshi)"
        )

    if get_rank() == 0:
        assert moshi_path is not None, "moshi_path must be provided for rank 0"
        assert _is_safetensors(moshi_path), f"Model is not safetensors: {moshi_path}"
        model_state_dict = safetensors.torch.load_file(str(moshi_path))

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            model_state_dict[k] = v.to(param_dtype)

        model.load_state_dict(model_state_dict, strict=False, assign=True)

        assert not any(p.is_meta for p in model.parameters()), (
            "All parameters should be initialized by now"
        )
        assert all(p.dtype == param_dtype for p in model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in model.parameters()), (
            "All parameters should be on meta"
        )

    torch.distributed.barrier()

    for param in model.parameters():
        param.requires_grad = True

    train_model = _TrainWrapper(model)

    if get_world_size() == 1:
        return train_model.cuda()

    auto_wrap_policy = get_fsdp_policy()

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        train_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
    )

    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)

    return wrapped_model
