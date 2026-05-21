import logging
import os
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str | None = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str | None = None
    moshi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None and self.config_path is None:
            print(
                "Warning: `hf_repo_id` is set but `config_path` is None. "
                "This will load default models."
            )


# ── PersonaPlex System Prompt 配置 ──────────────────────────────────────────
# 参考: PersonaPlex paper (arXiv:2602.06053), Section 3.1 Hybrid System Prompt
@dataclass
class SystemPromptConfig(Serializable):
    """PersonaPlex 的 Hybrid System Prompt 前缀配置。
    在对话 token 序列前注入:
      [Voice Prompt段] → [Silence] → [Text Prompt段] → [Silence] → [对话内容]
    每个段内三通道分配:
      - user audio 通道:   440Hz 正弦波
      - agent text 通道:   PAD(3) 或角色描述 tokens
      - agent audio 通道:  说话人音频 / 静音
    """
    # 启用/禁用 system prompt 前缀
    enable: bool = False
    # 每段静音缓冲时长（秒），默认 0.5s
    silence_duration_sec: float = 0.5
    # 默认文本角色描述，当 jsonl 中未指定 text_prompt 时使用
    default_text_prompt: str = ""
    # voice prompt 音频文件目录（可选，jsonl 会覆盖此值）
    voice_prompt_dir: str = ""


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5

    # ── PersonaPlex: text/audio loss 平衡系数 ─────────────────────────────
    # 原版 moshi-finetune 中 first_codebook_weight_multiplier=100 使 audio loss
    # 在数值上是 text loss 的 40-100 倍，导致微调后模型学不到精确的文本切换时机。
    # 此参数在总 loss 中为 text_loss 提供乘数补偿：
    #   mb_loss = text_loss * text_loss_weight + audio_loss
    # 建议值: 10-20（具体数值取决于 first_codebook_weight_multiplier 和
    # text_padding_weight 的设置）
    text_loss_weight: float = 1.0

    # ── PersonaPlex: Hybrid System Prompt 配置 ───────────────────────────
    system_prompt: SystemPromptConfig = field(default_factory=SystemPromptConfig)

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    duration_sec: float = 10
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If False, no checkpoints will be saved. This is useful for development.
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not
    # during the training process. Gradient checkpointing can be beneficial in
    # reducing memory usage at the cost of slightly longer training times.
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False

    param_dtype: str = "bfloat16"

    overwrite_run_dir: bool = False

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )
