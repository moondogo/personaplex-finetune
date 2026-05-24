import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
import torch
from moshi.models.lm import LMModel
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states"""

    def __init__(
        self,
        model: FullyShardedDataParallel | LMModel,
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        full_finetuning: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep
        self.full_finetuning = full_finetuning
        self.config = config

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(ckpt_dir: Path) -> Path:
        return ckpt_dir / "consolidated.safetensors"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "config.json"
        with open(params_path, "w") as f:
            f.write(json.dumps(self.config, indent=4))

    @torch.no_grad()
    def retrieve_save_states(
        self, save_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        offload_to_cpu = get_world_size() > 1

        assert (
            isinstance(self.model, FullyShardedDataParallel)
            or get_world_size() == 1
        ), (
            "`self.model` should be an instance of "
            "`FullyShardedDataParallel` if `world_size > 1`"
        )
        if get_world_size() > 1:
            with self.model.summon_full_params(
                writeback=True, offload_to_cpu=offload_to_cpu
            ):
                states = {k: v.to(dtype=save_dtype)
                          for k, v in self.model.state_dict().items()}
        else:
            states = {k: v.clone().to(dtype=save_dtype)
                      for k, v in self.model.state_dict().items()}

        states = dict(sorted(states.items()))
        return states

    @torch.no_grad()
    def save_checkpoint(
        self,
        dtype: torch.dtype = torch.float16,
    ):
        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(
            f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}"
        )

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            states: dict[str, torch.Tensor] = self.retrieve_save_states(dtype)

        barrier()

        if self.rank == 0:
            safetensors.torch.save_file(
                states,
                self.consolidated_path(tmp_dst),
            )
            self.write_params_info(tmp_dst)
            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}"
            )

            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
