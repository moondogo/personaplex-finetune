"""
PersonaPlex-finetune: 扩展现有 interleaver，支持 Hybrid System Prompt 前缀。

核心改动（相对于 moshi-finetune）：
1. 新增 build_hybrid_prefix()：构造 17 通道的 system prompt 前缀序列
   - 使用 PersonaPlex 硬编码的 SINE_TOKENS / SILENCE_TOKENS（适配其 Mimi 权重）
2. Sample/Batch 新增 system_prompt_len 字段，用于训练时 loss mask
3. Interleaver 新增 system prompt 相关参数
4. InterleavedTokenizer.__call__ 支持生成 17 通道序列（1 text + 8 agent audio + 8 user audio）

参考: PersonaPlex paper (arXiv:2602.06053), Section 3.1
"""
import json
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
from moshi.conditioners import ConditionAttributes

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]

# ── PersonaPlex: 硬编码的 Mimi token ────────────────────────────────────────
# 这些 token 是对 440Hz 正弦波和静音帧通过 PersonaPlex 的 Mimi 编码器得到的。
# 值从 personaplex/models/lm.py:56-57 搬来。如果你使用不同的 Mimi 权重，
# 需要重新编码并替换这些常量。
# shape: 每项是 [8] (1 semantic + 7 acoustic codebooks)
SILENCE_TOKENS = torch.tensor(
    [948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=torch.long
)  # 静音帧的 Mimi token
SINE_TOKENS = torch.tensor(
    [430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=torch.long
)  # 440Hz 正弦波的 Mimi token

# 帧率，与 PersonaPlex 保持一致
FRAME_RATE_HZ = 12.5
# 每个流（user/agent）的音频 codebook 数量
AUDIO_CODECOOKS_PER_STREAM = 8

# ── Special token 常量 ──────────────────────────────────────────────────────
# PersonaPlex LMModel 的 token 值与标准 Moshi 一致：
#   PAD=3, EPAD=0, ZERO=-1, BOS=1, EOS=2
PAD_TOKEN = 3
EPAD_TOKEN = 0
ZERO_TOKEN = -1


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    # ── PersonaPlex: system prompt 前缀帧数，用于 loss mask ──────────────
    # 如果此 sample 包含 system prompt 前缀，此值为前缀的总帧数。
    # 训练时此区域不回传 loss。None 表示无 system prompt。
    system_prompt_len: torch.Tensor | None = None


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    # ── PersonaPlex: 每个 sample 的 system prompt 前缀帧数 [B] ──────────
    system_prompt_len: torch.Tensor | None = None

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        # ── PersonaPlex: 收集各 sample 的 system_prompt_len ──────────
        prompt_lens = [b.system_prompt_len for b in batch]
        sys_len: torch.Tensor | None = None
        if any(pl is not None for pl in prompt_lens):
            lengths = []
            for pl in prompt_lens:
                if pl is None:
                    lengths.append(0)
                else:
                    lengths.append(pl.item() if isinstance(pl, torch.Tensor) else pl)
            sys_len = torch.tensor(lengths, dtype=torch.long)
        if batch[0].condition_attributes is None:
            return Batch(codes, system_prompt_len=sys_len)
        return Batch(codes, [b.condition_attributes for b in batch],
                     system_prompt_len=sys_len)


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


def encode_audio_frames(
    mimi,
    audio: np.ndarray,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """将 numpy 音频波形编码为 Mimi token 序列。
    返回 shape: [1, AUDIO_CODECOOKS_PER_STREAM, T]
    """
    tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    # tensor: [1, 1, T_samples]
    with torch.no_grad():
        codes = mimi.encode(tensor)  # [1, 8, T_frames]
    return codes


class Interleaver:
    """Interleaver with basic features
    Args:
        tokenizer: text tokenizer used by the model.
        audio_frame_rate (float): frame rate of the audio tokenizer.
        text_padding (int): special token used for text padding.
        end_of_text_padding (int): special token used to indicate end of text padding.
        zero_padding (int): special token id indicating that a 0 should be used instead
            of an actual embedding.
        in_word_padding (int | None): padding used within a word segment. Will default to `text_padding`.
        keep_main_only (bool): if True, will only keep the alignments with the main speaker.
        main_speaker_label (str): label for the main speaker in alignment data.
        use_bos_eos: (bool): if True, inserts BOS, EOS for change of turns.
        audio_delay (float): delay between the text and audio.
            A positive value means the text will be ahead of the audio.
        proba (float): probability of keeping the text.
        device: device location for the output tensors.
        ── PersonaPlex 新增参数 ──────────────────────────────────────────
        system_prompt_enable (bool): 是否启用 Hybrid System Prompt 前缀。
        silence_duration_sec (float): prompt 段间静音缓冲时长（秒），默认 0.5。
        voice_prompt_dir (str): voice prompt 音频文件目录。
        default_text_prompt (str): 默认角色描述，jsonl 未指定时使用。
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
        # ── PersonaPlex 新增 ──────────────────────────────────────────
        system_prompt_enable: bool = False,
        silence_duration_sec: float = 0.5,
        voice_prompt_dir: str = "",
        default_text_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device
        # ── PersonaPlex 新增 ──────────────────────────────────────────
        self.system_prompt_enable = system_prompt_enable
        self.silence_duration_sec = silence_duration_sec
        self.voice_prompt_dir = voice_prompt_dir
        self.default_text_prompt = default_text_prompt

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        # Tokenizes each word individually into a list of ints.
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments."""
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)

    # ═══════════════════════════════════════════════════════════════════════════
    # ── PersonaPlex: Hybrid System Prompt 前缀构建 ──────────────────────────
    # ═══════════════════════════════════════════════════════════════════════════

    def build_hybrid_prefix(
        self,
        voice_prompt_tokens: torch.Tensor,      # [1, 8, T_vp] — 预编码的 Mimi token
        text_prompt_token_ids: list[int],       # 角色描述 token ID 列表
    ) -> tuple[torch.Tensor, int]:
        """构建 PersonaPlex Hybrid System Prompt 前缀。

        结构（参考 paper Section 3.1）：
          [Voice Prompt 段] → [Silence 缓冲] → [Text Prompt 段] → [Silence 缓冲]

        每段的三通道分配：
                       text ch(0)    agent audio ch(1-8)    user audio ch(9-16)
          Voice Prompt: PAD (3)      说话人音频             440Hz 正弦波
          Silence:      PAD (3)      静音                  440Hz 正弦波
          Text Prompt:  角色描述     静音                  440Hz 正弦波

        Args:
            voice_prompt_tokens: 说话人音频的预编码 Mimi token [1, 8, T_vp]
            text_prompt_token_ids: 角色描述文本的 token ID 列表
        Returns:
            (prefix [1, 17, T_total], prefix_length) — 17通道前缀和总帧数
        """
        T_sil = max(1, int(self.silence_duration_sec * FRAME_RATE_HZ))

        # 构建单帧常量矩阵 [1, 8, 1]
        sine_1f = SINE_TOKENS.to(self.device).view(1, AUDIO_CODECOOKS_PER_STREAM, 1)
        sil_1f = SILENCE_TOKENS.to(self.device).view(1, AUDIO_CODECOOKS_PER_STREAM, 1)

        T_vp = voice_prompt_tokens.shape[-1]

        # ── Voice Prompt 段 ───────────────────────────────────────────────
        # text = PAD(3), agent audio = voice sample, user audio = sine
        vp_text = torch.full((1, 1, T_vp), PAD_TOKEN, device=self.device, dtype=torch.long)
        vp_user = sine_1f.expand(-1, -1, T_vp)  # 正弦波填满 user audio 通道
        vp = torch.cat([vp_text, voice_prompt_tokens, vp_user], dim=1)  # [1, 17, T_vp]

        # ── Silence 缓冲段 ────────────────────────────────────────────────
        sil_text = torch.full((1, 1, T_sil), PAD_TOKEN, device=self.device, dtype=torch.long)
        sil_agent = sil_1f.expand(-1, -1, T_sil)      # 静音
        sil_user = sine_1f.expand(-1, -1, T_sil)       # 正弦波
        sil = torch.cat([sil_text, sil_agent, sil_user], dim=1)  # [1, 17, T_sil]

        # ── Text Prompt 段 ────────────────────────────────────────────────
        # 角色描述文本 token 构成 text 通道
        if text_prompt_token_ids:
            T_tp = len(text_prompt_token_ids)
            tp_text = torch.tensor(text_prompt_token_ids, device=self.device,
                                   dtype=torch.long).view(1, 1, -1)
        else:
            # 无文本 prompt 时用单个 PAD 占位
            T_tp = 1
            tp_text = torch.full((1, 1, 1), PAD_TOKEN, device=self.device, dtype=torch.long)
        tp_agent = sil_1f.expand(-1, -1, T_tp)        # 静音
        tp_user = sine_1f.expand(-1, -1, T_tp)         # 正弦波
        tp = torch.cat([tp_text, tp_agent, tp_user], dim=1)  # [1, 17, T_tp]

        # ── 拼接：Voice Prompt → Silence → Text Prompt → Silence ────
        prefix = torch.cat([vp, sil, tp, sil], dim=2)  # [1, 17, T_prefix]
        total_frames = prefix.shape[-1]

        return prefix, total_frames

    def load_voice_prompt_tokens(
        self,
        voice_prompt_path: str,
        mimi,
    ) -> torch.Tensor | None:
        """加载 voice prompt 音频并编码为 Mimi token。

        支持两种格式：
        - .wav 文件：用 Mimi 编码为 token
        - .pt 文件：预编码的 token，直接从 'embeddings' key 恢复

        返回 shape: [1, 8, T_vp] 或 None（文件不存在时）
        """
        if not voice_prompt_path:
            return None
        vp_path = Path(voice_prompt_path)
        if self.voice_prompt_dir and not vp_path.is_absolute():
            vp_path = Path(self.voice_prompt_dir) / vp_path
        if not vp_path.exists():
            warnings.warn(f"Voice prompt file not found: {vp_path}")
            return None

        if vp_path.suffix == ".pt":
            # 预编码的 embedding 文件（PersonaPlex 格式）
            state = torch.load(str(vp_path), map_location="cpu")
            if "embeddings" in state:
                # .pt 文件中存储的是 transformer embeddings [T, dim]
                # 我们需要重新编码为 Mimi token — 这里暂不支持，
                # 请提供 .wav 文件或预编码的 token .pt 文件
                raise NotImplementedError(
                    "Voice prompt .pt embeddings replay not yet supported. "
                    "Please provide a .wav file instead."
                )
            elif "tokens" in state:
                return state["tokens"].to(self.device).view(1, AUDIO_CODECOOKS_PER_STREAM, -1)
            else:
                raise ValueError(f"Unrecognized .pt format for {vp_path}")
        else:
            # 编码 .wav 文件
            raw_audio, src_sample_rate = sphn.read(str(vp_path))  # (C, T)
            # ── 重采样到 Mimi 采样率（24kHz） ─────────────────────────
            mimi_sample_rate = getattr(mimi, 'sample_rate', 24000)
            if src_sample_rate != mimi_sample_rate:
                raw_audio = sphn.resample(
                    raw_audio,
                    src_sample_rate=src_sample_rate,
                    dst_sample_rate=mimi_sample_rate,
                )
            return encode_audio_frames(mimi, raw_audio, self.device)

    def tokenize_text_prompt(
        self,
        text_prompt_str: str,
    ) -> list[int]:
        """将文本角色描述 tokenize 为 ID 列表。

        PersonaPlex server.py 使用 wrap_with_system_tags 包裹角色描述：
          例如: "<system>You are a friendly bank teller.</system>"
        这里直接对原始字符串做 tokenize。
        """
        if not text_prompt_str:
            return []
        return tokenize(self.tokenizer, text_prompt_str, bos=False)

    def prepare_dialog_17ch(
        self,
        text_tokens: torch.Tensor,         # [1, 1, T] — 文本 token 流
        audio_tokens: torch.Tensor,        # [1, 8, T] — agent audio token
    ) -> torch.Tensor:
        """将对话段构造成 17 通道序列。

        结构: [text(1), agent_audio(8), user_audio(8)] = 17 channels
        其中 user audio 通道 (索引 9-16) 填充 ZERO_TOKEN (-1)，
        表示这些 token 在训练时来自外部输入（teacher forcing 上下文），
        不由模型预测。

        返回 shape: [1, 17, T]
        """
        T = text_tokens.shape[-1]
        # user audio 通道填 -1（zero_token），在训练对话段中作为外部输入
        user_audio = torch.full(
            (1, AUDIO_CODECOOKS_PER_STREAM, T),
            ZERO_TOKEN,
            device=text_tokens.device,
            dtype=torch.long,
        )
        return torch.cat([text_tokens, audio_tokens, user_audio], dim=1)  # [1, 17, T]


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


class InterleavedTokenizer:
    """将音频 + 对齐数据编码为多通道 token 序列。

    PersonaPlex 扩展：
    - 支持 17 通道输出（1 text + 8 agent audio + 8 user audio）
    - 支持在对话序列前插入 Hybrid System Prompt 前缀
    """

    def __init__(self, mimi, interleaver: Interleaver, duration_sec: float):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        # ── PersonaPlex: 是否启用 system prompt ──────────────────────
        self.has_system_prompt = interleaver.system_prompt_enable

    def __call__(self, wav: np.ndarray, start_sec: float, path: str,
                 voice_prompt: str | None = None,
                 text_prompt: str | None = None) -> Sample:
        """主入口：编码一段音频 + 对齐数据为 token 序列。

        PersonaPlex 扩展：
        - voice_prompt: jsonl 中的语音样本路径（.wav 或 .pt）
        - text_prompt: jsonl 中的角色描述文本
        - 在对话段前拼接 17 通道 system prompt 前缀
        """
        with torch.no_grad():
            audio_tensor = torch.Tensor(wav).cuda()
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
            audio_tokens = audio_tokens[..., : self.num_audio_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            info_file = os.path.splitext(path)[0] + ".json"
            with open(info_file) as f:
                data = json.load(f)
                alignments = data["alignments"]

            start_alignment = dicho(alignments, start_sec)
            end_alignment = dicho(alignments, start_sec + self.duration_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]

            text_tokens = self.interleaver.prepare_item(
                alignments, this_num_audio_frames
            )
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            # ── PersonaPlex: 回退到 Interleaver 默认文本 prompt ──────
            if text_prompt is None and self.interleaver.default_text_prompt:
                text_prompt = self.interleaver.default_text_prompt

            # ── PersonaPlex: 构建 system prompt 前缀 ──────────────────
            system_prompt_len: int = 0
            if self.has_system_prompt and voice_prompt is not None:
                # 加载并编码 voice prompt 音频
                voice_tokens = self.interleaver.load_voice_prompt_tokens(
                    voice_prompt, self.mimi
                )
                # tokenize 文本角色描述
                text_prompt_ids = self.interleaver.tokenize_text_prompt(
                    text_prompt or ""
                )

                if voice_tokens is not None:
                    prefix, prefix_len = self.interleaver.build_hybrid_prefix(
                        voice_tokens, text_prompt_ids,
                    )
                    prefix = prefix.to(device=text_tokens.device)

                    # 对话段扩展为 17 通道
                    dialog = self.interleaver.prepare_dialog_17ch(
                        text_tokens, audio_tokens,
                    )
                    codes = torch.cat([prefix, dialog], dim=2)  # [1, 17, T_total]
                    system_prompt_len = prefix_len

                    # 如果超出 num_audio_frames，截断（对话段被裁减）
                    if codes.shape[-1] > self.num_audio_frames:
                        codes = codes[..., : self.num_audio_frames]
                else:
                    # voice prompt 文件不可用，回退到无 prefix 模式
                    codes = self.interleaver.prepare_dialog_17ch(
                        text_tokens, audio_tokens,
                    )
            else:
                # ── 无 system prompt: 仍输出 17 通道（兼容 PersonaPlex 模型）──
                codes = self.interleaver.prepare_dialog_17ch(
                    text_tokens, audio_tokens,
                )

            sys_len_tensor = torch.tensor([system_prompt_len], dtype=torch.long)

            return Sample(
                codes,
                data.get("text_conditions", None),
                system_prompt_len=sys_len_tensor,
            )
