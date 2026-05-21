import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with  "path" and "duration" fields for each audio .wav file.

    PersonaPlex 扩展：jsonl 每行可额外包含:
      - voice_prompt: str  — 说话人语音样本路径（.wav 或 .pt）
      - text_prompt: str   — 角色描述文本（如 "<system>You are a bank teller.</system>"）
    如果 jsonl 未指定这些字段，将回退到 train args 中 system_prompt 的默认值。
    """

    train_data: str = ""
    shuffle: bool = False
    eval_data: str = ""
