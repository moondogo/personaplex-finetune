# PersonaPlex Finetune 训练数据管线详解

本文档完整描述训练数据从磁盘文件到模型输入的整个流程，包括每条 tensor 的精确 shape。

---

## 1. 数据流总览

```
JSONL manifest ──┬──► sphn.dataset_jsonl ──► stereo PCM chunks
(每行一个WAV)    │                            [2, T_samples]
                 │
.json 对齐文件 ──┘                            start_sec, path


                      ▼
            InterleavedTokenizer.__call__()
              │
              ├── Mimi.encode(agent channel) ──► agent audio tokens [1,8,T]
              ├── 读取 .json alignments ──► build_token_stream ──► text tokens [1,1,T]
              │
              ├── [可选] build_hybrid_prefix() ──► system prompt [1,17,T_prefix]
              │       (voice .wav → Mimi → [1,8,T_vp]
              │        text prompt → SentencePiece → [T_tp])
              │
              └── prepare_dialog_17ch() ──► codes [1,17,T_total]
                         │
                    Sample(codes, system_prompt_len)


                      ▼
              Batch.collate([Sample, ...])
                         │
                    Batch(codes=[B,17,T], system_prompt_len=[B])


                      ▼
              model.forward_train(codes) ──► LMOutput(logits, mask, text_logits, text_mask)
```

---

## 2. 数据源格式

### 2.1 JSONL 清单文件

每行一个 JSON 对象，描述一段训练音频：

```json
{
  "path": "/data/conv_001.wav",
  "duration": 300.0,
  "voice_prompt": "speaker_01.wav",
  "text_prompt": "<system>You are a friendly bank teller.</system>"
}
```

| 字段 | 必需 | 说明 |
|---|---|---|
| `path` | 是 | stereo WAV 文件路径，左声道=agent，右声道=user |
| `duration` | 是 | 音频总时长（秒）|
| `voice_prompt` | 否 | PersonaPlex voice prompt 音频路径（.wav 或 .pt），相对于 `voice_prompt_dir` |
| `text_prompt` | 否 | PersonaPlex 角色描述文本，未指定时回退到 `system_prompt.default_text_prompt` |

### 2.2 Stereo WAV 文件

- **左声道 (channel 0)**: Agent 音频（模型需要生成的输出）
- **右声道 (channel 1)**: User 音频（模型的外部输入，teacher-forcing 上下文）
- 采样率: 由 Mimi 模型决定（通常 24000 Hz）
- 格式: 通过 `sphn.dataset_jsonl` 加载为 numpy `[2, T_samples]`

### 2.3 对齐文件 (`.json`)

每个 WAV 对应一个同名的 `.json` 文件，包含时间戳对齐：

```json
{
  "alignments": [
    ["hello",  [0.0, 0.5],  "SPEAKER_MAIN"],
    ["world",  [0.5, 1.2],  "SPEAKER_MAIN"]
  ]
}
```

每条 alignment: `[word_text, (start_sec, end_sec), speaker_label]`

### 2.4 音频分片策略

JSONL 会根据 `duration_sec` 将长音频切分为固定长度的时间窗：

```python
# dataset.py: maybe_load_local_dataset
start_sec = 0
while start_sec < data["duration"]:
    chunks.append((data["path"], start_sec))
    start_sec += duration  # instruct_tokenizer.duration_sec
```

每个 chunk 通过 `sphn.dataset_jsonl` 读取对应的 stereo PCM 片段。

---

## 3. 数据集加载

### 3.1 入口: `build_data_loader()`

```python
# data_loader.py
dataset = build_dataset(
    pretrain_data=args.train_data,   # e.g. "/data/conv.jsonl"
    instruct_tokenizer=interleaved_tokenizer,
    seed=args.seed, rank=rank, world_size=world_size,
    is_eval=False, shuffle_pretrain=args.shuffle,
)

# 按 batch_size 聚合为 Batch
for sample in dataset:
    sample_list.append(sample)
    if len(sample_list) == batch_size:
        yield Batch.collate(sample_list)
```

### 3.2 DDP 分片

```python
# dataset.py: load_file
for idx, line in enumerate(f):
    if not idx % world_size == rank:  # 按 rank 轮询
        continue
    lines.append(line)
```

每个 GPU rank 只加载自己负责的 JSONL 行，通过 `sphn.dataset_jsonl(seq(skip=rank, step_by=world_size))` 进一步保证不重复。

### 3.3 多数据集采样

支持 `path1:weight1,path2:weight2` 格式配置多个数据源，训练时按权重随机交替采样：

```python
# dataset.py: interleave_iterators
while True:
    it_id = rng.choice(range(len(iterators)), p=probabilities)
    yield next(iterators[it_id])
```

### 3.4 Voice Prompt / Text Prompt 预读

由于 `sphn.dataset_jsonl` 可能不传递 JSONL 的额外字段，`_load_jsonl_extra_fields()` 预先读取 JSONL 建立 path→{voice_prompt, text_prompt} 的查找表作为 fallback。

---

## 4. Token 化详解

核心在 `InterleavedTokenizer.__call__()`。输入为一段 stereo PCM + 对齐数据，输出 17 通道 token 序列。

### 4.1 文本对齐 → Text Token Stream

**步骤**:

```
alignment JSON  ──► prepare_item() ──► build_token_stream() ──► [1, 1, T]
```

1. `dicho(alignments, start_sec)` 二分查找起始对齐位置
2. 提取 `[start_sec, start_sec+duration_sec]` 内的 alignment
3. 每个 word 通过 `SentencePiece.tokenize()` 转为 token ID
4. 按开始时间将 token **逐一分配到对应帧位置**

**`build_token_stream` 算法** (逐帧填充):

```
T = ceil(segment_duration * frame_rate)   # frame_rate = 12.5 Hz

初始化: text_tokens = [PAD] * T

for t in range(T):
    # 将当前帧之前开始的 word 的 tokens 压入队列
    while alignments[i].start * frame_rate < t + 1:
        to_append_stack = deque(alignments[i].tokens)
        i += 1

    if to_append_stack:
        if 上一帧是 PAD:
            text_tokens[t-1] = EPAD   # 标记 padding 结束
        text_tokens[t] = to_append_stack.popleft()  # 每帧放一个 token
    elif t <= last_word_end:
        text_tokens[t] = IN_WORD_PAD   # 词内填充
```

**关键约定**:
- 每个 80ms 帧最多承载一个 text token
- PAD(3) 填充无词区域，EPAD(0) 标记填充段结束
- IN_WORD_PAD 标记词内还在持续

### 4.2 Agent 音频 → Audio Token Stream

```python
# interleaver.py:509-520
audio_tensor = torch.Tensor(wav).cuda()          # [2, T_samples] (stereo)
agent_channel = audio_tensor[0:1, :]            # [1, T_samples] 只取左声道
audio_tokens = self.mimi.encode(agent_channel[:, None])
# Mimi.encode: [1, 1, T_samples] → [1, 8, T_frames]
#   SplitRVQ: rvq_first(1 semantic) + rvq_rest(7 acoustic) = 8 codebooks
```

**Shape 变换**:
| 步骤 | Shape | 说明 |
|---|---|---|
| `mimi.encode(input)` | `[1, 8, T_frames]` | 8 codebooks, T_frames ≈ T_samples / 1920 |
| `[..., :num_audio_frames]` | `[1, 8, num_audio_frames]` | 截断至目标帧数 |
| `F.pad(audio, (0, pad))` | `[1, 8, num_audio_frames]` | 零填充不足帧数 |
| `view(1, -1, T)` | `[1, 8, T]` | 保持 channel 维度 |

### 4.3 System Prompt 前缀 (可选)

当 `system_prompt.enable=True` 且 `voice_prompt` 字段存在时，构建 Hybrid System Prompt 前缀。

**结构** (PAPER Section 3.1):

```
┌─ Voice Prompt ─┬─ Silence ─┬─ Text Prompt ─┬─ Silence ─┐
│  text: PAD(3)  │  PAD(3)   │  角色描述文本  │  PAD(3)    │  ← dim=0 (text)
│  agent: 说话人 │  静音     │  静音          │  静音      │  ← dim=1-8 (agent audio)
│  user:  440Hz  │  440Hz    │  440Hz         │  440Hz     │  ← dim=9-16(user audio)
└────────────────┴───────────┴────────────────┴────────────┘
```

**硬编码 Mimi tokens** (来自 `personaplex/models/lm.py`):

```python
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]  # 静音帧
SINE_TOKENS    = [430, 1268, 381, 1611, 1095, 1495, 56, 472]    # 440Hz 正弦波
```

每个是 8 元素的一帧 token。

**Voice Prompt 加载**:
- `.wav` 文件: 通过 Mimi 编码 → `[1, 8, T_vp]`
- `.pt` 文件: 直接恢复 `tokens` key → `[1, 8, T_vp]`

### 4.4 通道组装

```python
# prepare_dialog_17ch()
text_tokens   = [1, 1, T]      # dim 0: text
agent_audio   = [1, 8, T]      # dim 1-8: agent audio (Mimi encoded)
user_audio    = [1, 8, T]      # dim 9-16: ZERO_TOKEN (-1) teacher-forcing slot

codes = torch.cat([text_tokens, agent_audio, user_audio], dim=1)
# → [1, 17, T]
```

**17 通道布局**:

| 索引 | 内容 | 预测者 | training target |
|---|---|---|---|
| `0` | text token | 主 Transformer | 对齐标注的 word token |
| `1` | agent semantic | Depformer cb=0 | Mimi token (semantic codebook) |
| `2-8` | agent acoustic | Depformer cb=1..7 | Mimi token (acoustic codebook) |
| `9` | user semantic | Depformer cb=8 | ZERO_TOKEN (-1, loss masked) |
| `10-16` | user acoustic | Depformer cb=9..15 | ZERO_TOKEN (-1, loss masked) |

**Silence 缓冲时长**: `silence_duration_sec * FRAME_RATE_HZ` 帧 → 默认 0.5s → 6 帧。

### 4.5 完整 Shape 规格表

| 中间数据 | Shape | 说明 |
|---|---|---|
| `wav` (stereo) | `[2, T_samples]` | 原始 PCM |
| `agent_channel` | `[1, T_samples]` | 左声道 |
| `mimi.encode(input)` | `[1, 8, T_frames]` | 8 codebook, B=1 |
| `audio_tokens` | `[1, 8, num_audio_frames]` | 截断+填充后 |
| `text_tokens` | `[1, 1, num_audio_frames]` | 文本 token stream |
| **dialog** (无 prefix) | `[1, 17, num_audio_frames]` | 对话段 |
| **prefix** | `[1, 17, T_prefix]` | System prompt 前缀 |
| **codes** (完整) | `[1, 17, T_total]` | 最终 codes，可能 ≤ num_audio_frames |
| **Batch.codes** | `[B, 17, T]` | B = batch_size, 沿 dim=0 cat |
| **system_prompt_len** | `[B]` | 每个 sample 的 prefix 帧数 |

---

## 5. 批处理

```python
# Batch.collate (interleaver.py:71-85)
codes = torch.cat([b.codes for b in batch])        # [B, 17, T]
prompt_lens = [b.system_prompt_len for b in batch]  # [B]
return Batch(codes, system_prompt_len=sys_len)
```

`Sample.codes` 的 `dim=0` 始终为 1（单 sample），直接沿 `dim=0` 拼接为 batch。

---

## 6. 训练循环中的消费

### 6.1 模型前向

```python
# train.py
batch = next(data_loader)
codes = batch.codes           # [B, 17, T]
output = model.forward_train(codes)
```

`forward_train` 内部 (lm.py:531-552):
1. `_delay_sequence(delays, codes, initial_tokens)` — 每条 codebook 按 `delays[k]` 偏移
2. `forward_codes(delayed[:, :, :-1])` — 主 Transformer 前向
3. `forward_depformer_training(delayed[:, :, 1:], transformer_out)` — Depformer 前向
4. `_undelay_sequence(delays[1:17], logits)` — 将 logits 对齐回原始位置

### 6.2 Loss 计算

```python
# text loss
text_loss = compute_loss_with_mask(
    output.text_logits,          # [B, 1, T, text_card]
    codes[:, :model.audio_offset],  # codes[:, 0:1, :] — text channel
    text_mask,
    mode="text",
    text_padding_weight=args.text_padding_weight,
    text_padding_ids={model.text_padding_token_id, model.end_of_text_padding_id},
)

# audio loss
audio_loss = compute_loss_with_mask(
    output.logits,               # [B, 16, T, card]
    codes[:, audio_offset:audio_offset+dep_q],  # codes[:, 1:17, :] — audio channels
    audio_mask,
    mode="audio",
    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
)

mb_loss = text_loss * args.text_loss_weight + audio_loss
```

**Loss mask 逻辑** (`compute_loss_with_mask`):
- `target_mask` 中 `False` 的位置被跳过（不参与 loss 计算）
- `first_codebook_weight_multiplier`: semantic codebook 的 loss 权重倍数
- `text_padding_weight`: PAD/EPAD token 的 loss 权重调节

### 6.3 System Prompt Loss Mask

```python
# train.py:257-270
if batch.system_prompt_len is not None:
    prompt_mask = torch.ones(B, T, dtype=torch.bool)
    for b in range(B):
        plen = batch.system_prompt_len[b].item()
        if plen > 0:
            prompt_mask[b, :plen] = False     # prefix 区域不回传 loss

    audio_mask = output.mask & prompt_mask[:, None, :]   # [B, 16, T]
    text_mask_ = output.text_mask & prompt_mask[:, None, :]  # [B, 1, T]
```

与 PersonaPlex Paper Section 3.1 一致: "During training, we mask out loss backpropagation to the system prompt."

---

## 7. 跨版本对比：原版 Moshi vs PersonaPlex

### 7.1 原版 moshi-finetune interleaver

```python
# 原版: from moshi-finetune/finetune/data/interleaver.py
audio_tensor = torch.Tensor(wav).cuda()           # [2, T]
audio_tokens = self.mimi.encode(audio_tensor[:, None])
# [:, None] → [2, 1, T] → Mimi 以 B=2 处理双声道
# Mimi.encode 返回 [2, 8, T_frames]

audio_tokens = audio_tokens.view(1, -1, T)
# view(1, -1, T) → [1, 16, T]   ← 合并两个声道

codes = torch.cat([text_tokens, audio_tokens], dim=1)
# → [1, 17, T]  (1 text + 16 audio)
```

**原版意图**: 故意利用 `[:, None]` 的 batch trick 同时编码两个声道，产生 16 个真实 audio token。模型 `n_q=16`, `dep_q=8` — depformer 仅预测前 8 个 (agent)，后 8 个 (user) 作为 teacher-forcing 上下文。

### 7.2 PersonaPlex-finetune interleaver (当前)

```python
# 修复后:
agent_channel = audio_tensor[0:1, :]              # 只取左声道的 agent
audio_tokens = self.mimi.encode(agent_channel[:, None])
# → [1, 8, T_frames]

codes = prepare_dialog_17ch(text_tokens, audio_tokens)
# → [1, 17, T]  (1 text + 8 agent + 8 ZERO_TOKEN user)
```

**差异**:

| | 原版 | PersonaPlex |
|---|---|---|
| Agent audio tokens | 左声道 8 codebook | 左声道 8 codebook |
| User audio tokens | 右声道 **真实** Mimi token | **ZERO_TOKEN** (-1) |
| 训练语义 | user audio 提供真实上下文 | user audio loss 被 mask |
| 导致 (17,25) bug | `[B,17,T]` 正常 | `[B,25,T]` 崩溃（修复前）|

### 7.3 (17, 25) Bug 根因

修复前代码使用 `audio_tensor[:, None]`，产生 `[2, 8, T_frames]`。`view(1, -1, T)` 将其展平为 `[1, 16, T]`，再经 `prepare_dialog_17ch` 加上 8 个 ZERO_TOKEN user 通道 → `[1, 25, T]`。

模型期望 `num_codebooks=17`（`n_q=16`），收到 `K=25` 导致 `_delay_sequence` assertion 失败。

**修复**: `audio_tensor[0:1, :]` 只取 agent 声道 → Mimi 返回 `[1, 8, T]` → 最终 `[1, 17, T]` ✓

---

## 8. Special Token 速查表

### 8.1 控制 token

| Token | ID | 用途 |
|---|---|---|
| `EPAD` | 0 | End-of-padding：标记文本填充结束 |
| `BOS` | 1 | Beginning-of-sentence |
| `EOS` | 2 | End-of-sentence |
| `PAD` | 3 | 文本填充 token |
| `ZERO` | -1 | 特殊输入值：模型不对该位置采样，embedding 输出 0 |
| `UNGENERATED` | -2 | 提示该位置应由模型生成 |

### 8.2 PersonaPlex 硬编码 Mimi token (per frame)

| Token 组 | 值 (8 codebooks) | 用途 |
|---|---|---|
| `SILENCE_TOKENS` | `[948, 243, 1178, 546, 1736, 1030, 1978, 2008]` | 静音帧的 Mimi code |
| `SINE_TOKENS` | `[430, 1268, 381, 1611, 1095, 1495, 56, 472]` | 440Hz 正弦波的 Mimi code |

**注意**: 这些值是针对特定 PersonaPlex Mimi 权重 (tokenizer-e351c8d8-checkpoint125.safetensors) 编码得到的。更换 Mimi 权重需要重新编码。

---

## 9. 关键常量

| 常量 | 值 | 来源 |
|---|---|---|
| `AUDIO_CODECOOKS_PER_STREAM` | 8 | interleaver.py:46 |
| `FRAME_RATE_HZ` | 12.5 | interleaver.py:44 |
| `SAMPLE_RATE` | 24000 | loaders.py:39 |
| `frame_size` | 1920 samples (24000/12.5) | Mimi 计算 |
| `n_q` (模型) | 16 | loaders.py `_lm_kwargs` |
| `dep_q` (模型) | 16 | loaders.py `get_moshi_lm` 覆盖 |
| `num_codebooks` | 17 (= n_q + 1) | LMModel 计算 |
| `len(delays)` | 17 (= num_codebooks) | 不变式 |
| `num_audio_frames` | `ceil(duration_sec * 12.5)` | InterleavedTokenizer |

---

## 10. 配置参数影响

### Interleaver 参数

| 参数 | 默认 | 影响 |
|---|---|---|
| `keep_main_only` | True | 只保留主说话人的文本 token |
| `system_prompt_enable` | 取决于 yaml | 是否在对话前插入 Hybrid System Prompt |
| `silence_duration_sec` | 0.5 | System Prompt 段间静音缓冲 |
| `voice_prompt_dir` | "" | Voice prompt 文件目录 |
| `default_text_prompt` | "" | JSONL 未指定时的默认角色描述 |

### 训练参数

| 参数 | 建议值 | 影响 |
|---|---|---|
| `duration_sec` | 164 | 每段训练序列的时长 (s) |
| `first_codebook_weight_multiplier` | 10 | Semantic codebook loss 倍数 |
| `text_padding_weight` | 1.0 | PAD/EPAD token loss 权重 |
| `text_loss_weight` | 20 | text loss 全局补偿系数 |
| `batch_size` | 32 | 每步训练样本数 |

---

## A. 单条 Sample 的完整 Shape 追踪 (以 `duration_sec=164` 为例)

```
sphn 加载 stereo WAV:
  wav.shape              = (2, 3932160)    # 164s × 24000Hz = 3936000 samples

取左声道:
  agent_channel.shape    = (1, 3932160)

Mimi 编码 (frame_rate=12.5Hz):
  emb.shape              = (1, 256, 2050)  # 3932160 / 1920 = 2048 frames → ceil=2050
  mimi.encode returns    = (1, 8, 2050)    # 8 codebooks × 2050 frames

截断至目标帧数 (num_audio_frames = ceil(164×12.5) = 2050):
  audio_tokens.shape     = (1, 8, 2050)

对齐文本:
  text_tokens.shape      = (1, 1, 2050)

组装 dialog:
  prepare_dialog_17ch():
    user_audio.shape     = (1, 8, 2050)    # ZERO_TOKEN-filled
    codes.shape          = (1, 17, 2050)

Batch (batch_size=4):
  codes.shape            = (4, 17, 2050)
  system_prompt_len      = tensor([0, 0, 0, 0])   # 无 system prompt

模型输入:
  forward_train(codes)   # codes = [4, 17, 2050]
    _delay_sequence(delays=17项, codes)
    → delayed_codes      # [4, 17, 2051] (含 initial token)
    forward_codes        # [4, 17, 2050] → text_logits [4, 1, 2050, text_card]
    forward_depformer    # [4, 17, 2051] → logits [4, 16, 2050, card]
    _undelay_sequence    # 对齐回原始时间轴

Loss:
  text_loss    (4, 1, 2050) → scalar
  audio_loss   (4, 16, 2050) → scalar
  total_loss = text_loss * 20 + audio_loss
```
