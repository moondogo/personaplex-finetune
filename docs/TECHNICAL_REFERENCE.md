# PersonaPlex-Finetune 技术参考文档

本文档汇总了基于 Moshi / PersonaPlex 论文和代码分析得出的关键技术结论，
以及 `personaplex-finetune` 项目的设计决策。

---

## 1. 模型架构对比

### Moshi (arXiv:2410.00037)

全双工语音对话模型。核心组件：

| 组件 | 说明 |
|---|---|
| **Helium** | 7B 文本 LLM，2.1T tokens 预训练 |
| **Mimi** | 流式神经音频编解码器。24kHz ↔ 离散 tokens，12.5Hz 帧率，80ms 帧大小。8 个码本（1 语义 + 7 声学），1.1kbps |
| **RQ-Transformer** | 双 Transformer 架构：大 Temporal Transformer（7B）+ 小 Depth Transformer（~6 层，dim=1024） |
| **Multi-stream** | 在一个联合序列中作为并行流对用户音频和 Moshi 音频建模，消除显式发言人轮次 |
| **Inner Monologue** | 在每个时间步先预测时间对齐文本 token，再预测音频 token。关键质量机制——口语问答正确率近三倍提升 |
| **Acoustic delay** | 语义和声学 token 偏移 1-2 个时间步。理论延迟 160ms，实际约 200ms |

**每时间步的联合序列结构**（delay=1）：
```
[user_semantic, user_acoustic₁…₇, moshi_text, moshi_semantic, moshi_acoustic₁…₇]
```

Token 由 Depth Transformer **自底向上**预测。

### PersonaPlex (arXiv:2602.06053)

基于 Moshi，增加角色条件控制和零样本声音克隆：

| 特性 | 说明 |
|---|---|
| **Hybrid System Prompt** | 将**语音提示段**（agent audio 通道放置短语音样本）+ **文本提示段**（在 agent text 通道强制角色描述 token）串联。提示期间用户音频替换为 440Hz 正弦波 |
| **三输入通道** | user audio / agent text / agent audio — 与 Moshi 的 multi-stream 相同 |
| **dep_q=16** | 8 个 agent audio 码本 + 8 个 user audio 码本均由深度 Transformer 生成 |
| **训练数据** | 合成对话，约 1840 小时服务类对话 + 410 小时 QA 对话。文本由 Qwen-3-32B / GPT-OSS-120B 生成，语音由 Dia / Chatterbox TTS 生成 |
| **训练参数** | 8×A100 GPU 约 6 小时，batch 32，最大 seq 2048 tokens（163.84 秒） |

---

## 2. Special Token 对比

Moshi 和 PersonaPlex 的 **special token 数值完全一致**：

| Token | 值 | 含义 |
|---|---|---|
| PAD（text_padding） | **3** | 文本流中的填充/静默 token |
| EPAD（end_of_text_padding） | **0** | 标记某词之前最后一次填充步骤 |
| ZERO | **-1** | 无需采样/跳过（嵌入输出零向量） |
| UNGENERATED | **-2** | 尚未生成 token 的占位符 |
| audio initial | `card`（2048） | 音频流起始 token |
| text initial | `text_card`（32000） | 文本流起始 token |

**关键差异不在 token 值，而在架构参数**：

| 参数 | Moshi | PersonaPlex |
|---|---|---|
| `dep_q` | 8 | **16** |
| `n_q` | 16 | 16 |
| `existing_text_end_padding_id` | 构造函数参数（默认 0） | 硬编码 `return 0` |
| `depformer_norms` | 有（每步独立 norm） | 无 |
| Condition 系统 | 有（ConditionProvider/Fuser） | 无 |
| `zero_text_code`（LMGen） | 无 | 硬编码 `= 3`（PAD），系统提示期间强制静音 |

---

## 3. 微调延迟问题根因分析

### 现象

使用 moshi-finetune 对 Moshi 模型进行 LoRA 微调后，模型能正常对话，但**有一个明显可感的延迟才开始响应**。
通过对照试验，排除服务端及网络延迟的可能性。

### 根因

**训练 loss 权重严重失衡**。

在原版 `moshi_7B.yaml` 中：

```yaml
first_codebook_weight_multiplier: 100.   # 语义音频码本 ×100
text_padding_weight: 0.5                 # PAD/EPAD token ×0.5
```

导致 `mb_loss = text_loss + audio_loss` 中：
- Audio 有 8 个码本 vs Text 仅 1 个
- 语义码本额外 ×100
- 文本流中大多数帧是 PAD，权重又减半

→ **audio loss 在数值上是 text loss 的 40-100 倍**

### 机制

Moshi 在每个时间步同时预测 text token（决定"说什么"）和 audio token（决定"如何发音"）。
文本 token 决定**何时开始说话**——模型预测 PAD(3) = 沉默，预测单词 = 说话。

EPAD(0) 作为**预示信号**：训练数据中 EPAD 出现在每个单词起始时间步的**前一帧**，教会模型"下一帧就有单词了"。

当 LoRA（rank=128）的有限适应能力几乎全部被音频优化消耗时：
- 模型学不到精确的**文本切换时机**（从 PAD 切到 EPAD 再到单词）
- 表现为在用户说完后多预测好多个 PAD(3) 才敢切到 EPAD(0)
- 听觉上：延迟

### 修复

三管齐下：

```yaml
# personaplex_finetune.yaml
first_codebook_weight_multiplier: 10.    # 100 → 10
text_padding_weight: 1.0                 # 0.5 → 1.0（不给 PAD/EPAD 降权）
text_loss_weight: 20.                    # 新增：text_loss * 20 补偿
```

对应 `train.py` 中：
```python
mb_loss = text_loss * args.text_loss_weight + audio_loss
```

---

## 4. Hybrid System Prompt 训练设计

### 原理

PersonaPlex 在对话序列前注入 system prompt 前缀，结构为：

```
[Voice Prompt 段] → [Silence 缓冲] → [Text Prompt 段] → [Silence 缓冲] → [对话内容]
```

每段内三通道分配：

| 通道 | Voice Prompt 段 | Text Prompt 段 | Silence 段 |
|---|---|---|---|
| **text** (ch 0) | PAD (3) | 角色描述 tokens | PAD (3) |
| **agent audio** (ch 1-8) | **说话人音频**（Mimi token） | 静音（SILENCE_TOKENS） | 静音 |
| **user audio** (ch 9-16) | 440Hz 正弦波（SINE_TOKENS） | 440Hz 正弦波 | 440Hz 正弦波 |

### 硬编码 Token

正弦波和静音的 Mimi token 是**针对特定 Mimi 权重预计算的**，存放在 `finetune/data/interleaver.py` 中：

```python
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]   # 8 码本
SINE_TOKENS    = [430, 1268, 381, 1611, 1095, 1495, 56, 472]     # 8 码本
```

这些值从 `personaplex/models/lm.py:56-57` 搬来。
**仅对 PersonaPlex 官方 Mimi 权重有效**。

### 训练时 Loss Mask

在训练期间，system prompt 区域**不回传 loss**（与 PersonaPlex paper Section 3.1 一致）：

```
"During training, we mask out loss backpropagation to the system prompt."
```

实现方式：`train.py` 在每个 batch 根据 `batch.system_prompt_len` 构建 `prompt_mask`，
与 `output.mask` / `output.text_mask` 按位与，该区域的 loss weight 为零。

### 对话段

对话段的 17 通道中，user audio 通道（9-16）填充 `ZERO_TOKEN (-1)`。
这意味着：
- **Temporal Transformer** 看到这些 token 作为上下文（teacher forcing），但不预测它们
- **Depth Transformer** 在训练时对 ZERO_TOKEN 位置的 loss 被 mask 掉
- 训练时 user audio 通道仅作为条件输入，模型学习对其做出反应

---

## 5. 17 通道架构

```
索引:  0    1  2  3  4  5  6  7  8    9 10 11 12 13 14 15 16
       ├─ text ─┤├──── agent audio (8) ────┤├──── user audio (8) ────┤
```

总通道数 = `n_q + 1 = 17`（1 text + 16 audio）

Delay 数组（PersonaPlex 与 Moshi 相同）：
```
[0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]
 text  agent audio 0-7  user audio 0-7
```

- 延迟=0 的 token（文本 + 语义码本）在时间步上对齐
- 延迟=1 的 token（声学码本）滞后一个时间步，减少码本间依赖

### 数据流

```
JSONL（含 voice_prompt、text_prompt 字段）
  │
  ├─ dataset.py: _load_jsonl_extra_fields() 预读到 lookup 表
  │    （fallback 机制：sphn.dataset_jsonl 可能不传递额外字段，
  │     因此独立提取 path→{voice_prompt, text_prompt} 映射）
  │
  ▼
InterleavedTokenizer.__call__()
  ├─ 若有 voice_prompt：
  │    1. load_voice_prompt_tokens() → sphn.read() + 重采样 + Mimi 编码
  │       （重采样到 24kHz，因为 voice prompt 可能来自不同采样率的音频）
  │    2. tokenize_text_prompt() → text prompt token 列表
  │    3. build_hybrid_prefix() → [1, 17, T_prefix]
  │    4. prepare_dialog_17ch() → [1, 17, T_dialog]
  │    5. torch.cat([prefix, dialog]) → [1, 17, T_total]
  │    6. system_prompt_len = T_prefix
  │
  └─ 若无 voice_prompt：
        prepare_dialog_17ch() → [1, 17, T]（无前缀）
  │
  ▼
Sample(codes=[1, 17, T], system_prompt_len=tensor)
  │
  ▼
Batch.collate() → Batch(codes=[B, 17, T], system_prompt_len=[B])
  │
  ▼
train.py → LMModel(codes) → LMOutput
  │
  ▼
Loss: mb_loss = text_loss * text_loss_weight + audio_loss
      （system prompt 区域 mask 掉）
```

---

## 6. 使用方法

### 环境

```bash
cd personaplex-finetune
uv run torchrun --nproc-per-node 8 -m train example/personaplex_finetune.yaml
```

或单 GPU：
```bash
uv run torchrun --nproc-per-node 1 -m train example/personaplex_finetune.yaml
```

### JSONL 数据格式

每行一个训练样本，最少包含：
```jsonl
{"path": "data/conversation.wav", "duration": 164}
```

PersonaPlex 扩展字段：
```jsonl
{
  "path": "data/conversation.wav",
  "duration": 164,
  "voice_prompt": "voices/spk001.wav",
  "text_prompt": "<system>You are a friendly bank teller.</system>"
}
```

- `voice_prompt`：说话人音频样本路径（.wav）。支持相对路径（基于 `system_prompt.voice_prompt_dir`）
- `text_prompt`：角色描述文本。若缺省则使用 `system_prompt.default_text_prompt`

### 训练配置关键参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `hf_repo_id` | `nvidia/personaplex-7b-v1` | PersonaPlex 官方权重 |
| `system_prompt.enable` | `true` | 启用 Hybrid System Prompt |
| `system_prompt.silence_duration_sec` | `0.5` | 提示段间静音缓冲 |
| `first_codebook_weight_multiplier` | `10` | 下调以避免延迟 |
| `text_padding_weight` | `1.0` | 不下调 PAD/EPAD 权重 |
| `text_loss_weight` | `20` | text loss 平衡系数 |
| `duration_sec` | `164` | 每段长度（秒）= 2048 tokens at 12.5Hz |
| `batch_size` | `32` | 遵循 PersonaPlex paper |
| `max_steps` | `24576` | 遵循 PersonaPlex paper |

### 关闭 System Prompt

若不希望注入 system prompt，设 `system_prompt.enable: false`。
此时每个 sample 仍输出 17 通道序列（兼容 PersonaPlex dep_q=16），但无前缀。

---

## 7. 与 moshi-finetune 的关键差异汇总

| 方面 | moshi-finetune 原版 | personaplex-finetune |
|---|---|---|
| 输出通道 | [1, 9, T]（1 text + 8 audio） | [1, 17, T]（1 text + 16 audio） |
| dep_q | 8 | 16（从 PersonaPlex 权重 config.json 读取） |
| 目标权重 | `text_loss + audio_loss`（audio 主导） | `text_loss * text_loss_weight + audio_loss` |
| first_codebook_weight | 100 | 10 |
| text_padding_weight | 0.5 | 1.0 |
| System Prompt | 无 | 可选 Hybrid System Prompt 前缀 |
| 正弦波/静音处理 | 无 | 硬编码 SINE_TOKENS / SILENCE_TOKENS |
| 语音样本 | 无 | Mimi 编码后注入 agent audio 通道 |
| 角色描述 | 无 | 注入 text 通道 |
| Loss mask | 仅 zero_token mask | 额外添加 system prompt 区域 mask |

---

## 8. 参考资料

- **Moshi paper**: [arXiv:2410.00037](https://arxiv.org/abs/2410.00037) — 全双工语音-文本基础模型
- **PersonaPlex paper**: [arXiv:2602.06053](https://arxiv.org/abs/2602.06053) — 全双工会话语音模型的语音和角色控制
- **moshi**: [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **moshi-finetune**: [github.com/kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)

---

## 9. 已知限制和注意事项

### sphn.dataset_jsonl 额外字段旁路

`sphn.dataset_jsonl` 可能不会将 jsonl 中的 `voice_prompt` / `text_prompt` 字段传递到
sample dict 中。为此 `dataset.py` 中添加了 `_load_jsonl_extra_fields()` 函数，
在迭代前预读 jsonl 构建 path→extra_fields 查找表。获取字段的优先级为：
1. 首先从 sphn 返回的 sample dict 获取（`sample.get("voice_prompt")`）
2. 若为 None，回退到预读的 lookup 表

如果需要完全绕过 sphn 的字段传递，直接在 jsonl 中写入 voice_prompt / text_prompt
字段即可——lookup 表会自动捕获。

### Voice Prompt 音频采样率

`load_voice_prompt_tokens()` 会自动将 voice prompt 音频重采样到 Mimi 的采样率（24kHz）。
使用 `sphn.resample()` 进行重采样。如果提供的是预编码 `.pt` 文件，则跳过此步骤。

### 硬编码 SINE_TOKENS / SILENCE_TOKENS

`finetune/data/interleaver.py` 中的硬编码 token 仅对 **PersonaPlex 官方 Mimi 权重**有效。
如果使用不同的 Mimi 权重或不同码本配置，需要使用目标 Mimi 编码器重新编码正弦波和静音帧，
并替换这些常量。

### 标准 Moshi LMModel 加载 PersonaPlex 权重

`personaplex-finetune` 使用标准 `moshi` 包的 `LMModel` 加载 PersonaPlex 权重。
标准 Moshi LMModel 比 PersonaPlex LMModel 多出一些参数（如 `depformer_norms`、`extra_heads` 等），
但通过 `load_state_dict(strict=False)` 加载权重时，这些额外参数保持默认值
（Identity 层，不影响计算）。`dep_q=16` 由 PersonaPlex 的 `config.json` 正确设置。

### lm_kwargs 配置传递

`train.py` 从 `checkpointer_info.raw_config` 获取 `lm_kwargs`。
PersonaPlex 的 `config.json` 会覆盖标准 Moshi 的默认值（包括 dep_q=16）。
如果 HF repo 的 config.json 不可用，会回退到 `_lm_kwargs`（dep_q=8），
导致权重加载不匹配。确保 `hf_repo_id` 指向有效的 PersonaPlex 权重仓库。

### System Prompt 段嵌入不会缓存

每次 `InterleavedTokenizer.__call__` 都会重新编码 voice prompt 音频。
对于大规模训练，建议在数据预处理阶段提前将 voice prompt 编码为 Mimi token，
存储为 `.pt` 文件（包含 `"tokens"` key），供 `load_voice_prompt_tokens()` 直接加载。
