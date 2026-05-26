# Moshi vs PersonaPlex Moshi: 架构与代码差异

本文档记录 PersonaPlex 对 Moshi 代码的全部改动，按改动类型分类。

---

## 一、架构级差异

### 1.1 移除 Classifier-Free Guidance (CFG)

PersonaPlex 完全移除了 CFG 推理机制：

| 组件 | Moshi 原版 | PersonaPlex |
|---|---|---|
| `conditioners/` 目录 | 存在（ConditionFuser, ConditionProvider, ConditionTensors） | **整个目录删除** |
| `LMModel.forward()` | 接受 `condition_tensors: ConditionTensors` 参数，内部处理 cross-attention + sum condition | **不存在**，改为 `forward_train(codes)` 纯自回归 |
| `LMGen` 构造函数 | 接受 `cfg_coef`, `condition_tensors` | 移除这些参数，改为 `device`, `sample_rate`, `frame_rate` |
| `LMGen.step()` | 使用 CFG mask 做双路推理 | 纯单路自回归采样 |
| `MHAState` | 包含 `k_cross`, `v_cross` 交叉注意力缓存 | 移除交叉注意力 |
| `_LMGenState` | 包含 `condition_sum`, `condition_cross`, `cfg_is_masked_until` | 全部移除 |

### 1.2 新增 Voice Prompt 系统

PersonaPlex 的核心创新——零样本语音克隆：

| 新增方法 | 功能 |
|---|---|
| `load_voice_prompt()` | 从 WAV 加载语音样本，编码为 Mimi tokens |
| `load_voice_prompt_embeddings()` | 从 .pt 文件加载预编码的 embeddings |
| `_encode_zero_frame()` | 编码静音帧 |
| `_encode_sine_frame()` | 编码 440Hz 正弦波帧（用于 user audio 通道占位） |
| `_step_voice_prompt_core()` | 将语音样本注入 agent audio 通道 |
| `_step_text_prompt_core()` | 将角色描述文本注入 agent text 通道 |
| `_step_audio_silence_core()` | 注入静音缓冲帧 |
| `step_system_prompts()` / `step_system_prompts_async()` | 编排三步流程：voice prompt → silence → text prompt → silence |

### 1.3 多通道架构变化

| 维度 | Moshi 原版 | PersonaPlex |
|---|---|---|
| 每步 token 流 | `[user_audio(8)]` + `[moshi_text(1) + moshi_audio(8)]` | `[text(1) + agent_audio(8) + user_audio(8)]` = **17 channels** |
| `dep_q` | 8 | **16** |
| Audio codebooks per stream | 8 | 8 × 2 (agent + user) |
| `audio_offset` | 1 | 1（不变） |

---

## 二、文件级差异（逐文件）

### 2.1 完全一致的文件（6 个）

`models/__init__.py`, `modules/__init__.py`, `quantization/__init__.py`, `quantization/base.py`, `utils/__init__.py`, `utils/autocast.py`

### 2.2 `__init__.py`

- **删除** `from . import conditioners`
- 版本号 `0.2.13` → `0.1.0`

### 2.3 `models/compression.py`

- `_MimiState` 不再继承 `State`，变为普通 dataclass
- **移除** `graphed_encoder` / `graphed_decoder`（CUDA Graph 包装）
- 新增 `torch_compile_encoder_decoder` 参数和 `_context_for_encoder_decoder` 上下文管理器
- **移除** `frame_size` property（两处）
- encoder/decoder 调用不再通过 graphed state，直接调用 `self.encoder(x)` / `self.decoder(x)`

### 2.4 `models/lm.py`（最大改动，~1600 行 diff）

**删除的功能**：
- 全部 CFG 机制：`cfg_coef`, `condition_tensors`, `cfg_is_masked_until`
- 全部 Conditioner 集成：`ConditionFuser`, `ConditionProvider`, `ConditionTensors`
- 量化支持：`replace_linear_with_qlinear`
- Extra heads：`extra_heads_num_heads`, `extra_heads_dim`
- Gradient checkpointing：`gradient_checkpointing` 参数
- `forward()` 方法（接受 codes + condition_tensors）

**新增的功能**：
- Voice prompt 全生命周期方法（15+ 个方法）
- `forward_train(codes)` 作为训练入口
- `forward_codes(sequence)` → `forward_embeddings(input)` 分步 API
- `embed_codes()` 从序列到 embeddings
- `multi_linear()` 替代 `apply_weights_per_step()`
- `ScaledEmbedding` 类内联到此文件
- `_delay_sequence()` / `_undelay_sequence()` 内联

**Streaming state 变化**：
- `_LMGenState` 不再继承 `State`，移除 `exec_mask`/`set_exec_mask` 逻辑
- Cache 大小 `max_delay + 2` → `max_delay + 3`
- 新增 `provided` tensor（teacher forcing 用）
- `LMGen.depformer_step()` 新增 `audio_tokens`/`audio_provided` 参数

### 2.5 `models/loaders.py`

**删除**：
- **整个 `CheckpointInfo` dataclass**（~120 行）
- `hf_get()`, `get_conditioner()`, `get_conditioner_provider()`, `get_condition_fuser()`, `get_lora_moshi()`
- 所有 LoRA、量化、条件器相关导入
- `MOSHI_Q8_NAME` 常量

**修改**：
- `DEFAULT_REPO`: `kyutai/moshiko-pytorch-bf16` → `nvidia/personaplex-7b-v1`
- `get_mimi()`: 文件名改为必需参数；`num_codebooks` 硬编码为 8；使用 `load_model()` 替代 `load_file()`
- `get_moshi_lm()`: `dep_q` 硬编码为 16；**新增权重补丁逻辑**：扩展 depformer self_attn 权重，复制 codebooks 0→7 到 8→15（针对 gating, linears, depformer_in, depformer_emb）
- **新增** `_get_moshi_lm_with_offload()`: 通过 `accelerate` 实现 CPU offload

### 2.6 `modules/conv.py`

- **完全重写** streaming 状态管理：从 overlap-add（`previous + first`）改为 padding-based 方案
- `nn.Conv1d` → `RawStreamingConv1d`；`nn.ConvTranspose1d` → `RawStreamingConvTranspose1d`
- 移除 `causal=True` 强制要求
- `pad_mode` 默认值 `"constant"` → `"reflect"`
- `trim_right_ratio` 从固定 `1.0` 改为可配置
- 删除 ~100 行的 `test()` 函数（移到 `streaming.py`）

### 2.7 `modules/streaming.py`（大改动）

- `State` 基类从带 `exec_mask`/`set_exec_mask` 的复杂 dataclass 改为 `Resetable` Protocol
- 移除 `exec_mask` 机制
- `_streaming_detached` 改为 `_streaming_propagate`（语义反转）
- 移除 `_cached_children` 缓存
- **新增** `save_streaming_state()`, `set_streaming_state_inplace()`, `load_streaming_state()` — 流状态序列化
- **新增** `StreamingAdd` — 不等长张量流式加法
- **新增** `RawStreamingConv1d` / `RawStreamingConvTranspose1d` — 基于 overlap-add 的流式卷积
- **新增** `safe_asdict()`, `_flatten_streaming_state()`, `_restore_streaming_state_pt()` 等辅助函数

### 2.8 `modules/transformer.py`

- **删除** `expand_repeated_kv()` (GQA 支持)
- **删除** `apply_weights_per_step()` (98 行)，被 `multi_linear()` 替代
- `StreamingMultiheadAttention`:
  - `in_projs`/`out_projs` ModuleList 改为单个 `nn.Linear`
  - 移除交叉注意力全部逻辑（`cross_attention`, `kv_repeat`, `cache_cross_attention`）
- `_MHAState` 不再继承 `State`；移除 `k_cross`/`v_cross`
- `RingKVCache` 移除 `respect_exec_mask`；集成 `asdict()` 方法
- `KVCacheResult.from_kv()` positions 不再 expand
- 移除 `_load_hook`（旧格式 ModuleList 权重加载）

### 2.9 `modules/gating.py`

- **删除** `gating_forward_generic()` 回退函数
- **删除** `quantized` 参数
- `forward()` 始终使用 `gating_forward_kernel()`，移除 `no_compile()` 分支

### 2.10 `modules/rope.py`

- **删除** `interleave` 参数（始终交错格式）
- 广播逻辑简化（不依赖 batch 维度 `B`）
- `RotaryEmbedding.__init__` 移除 `interleave` 参数

### 2.11 `modules/seanet.py`

- `SEANetResnetBlock`: `u + v` 改为 `self.add(u, v)`（`StreamingAdd`）
- `SEANetEncoder.forward()` 和 `SEANetDecoder.forward()` 添加 `@torch_compile_lazy` 装饰器

### 2.12 `modules/resample.py`

- 仅有 license 头变更，零代码改动

### 2.13 `quantization/core_vq.py`

**大量删除——训练逻辑全部移除**：
- `_run_kmeans()` — K-means 码本初始化（~40 行）
- `_average_tensors()` — 分布式同步
- `_init_embedding()` — 自初始化
- `_check_expired_codes()` — 过期码字替换
- `initialized` property
- EMA 跟踪（`cluster_usage`, `embedding_sum`）
- STE commit loss（`quantized = x + (quantized - x).detach()`）
- **结论**: 码本仅依赖预训练权重，不支持训练时自初始化

### 2.14 `quantization/vq.py`

- 移除 `q_dropout` 随机减少码本数逻辑
- 移除 `no_quantization_rate` 随机 mask 逻辑
- **新增** `no_quantization_mode` 参数（`"true_skip"` / `"same"` / `"independent"`）
- **新增** `generator_seed` + `torch.Generator` 替代 `random.Random`
- `SplitResidualVectorQuantizer` 支持 `no_quantization_mode`

### 2.15 `utils/compile.py`

- 移除 `isinstance(module_for_sig, torch.nn.Module)` 断言
- 新增 `CUDAGraphed.asdict()` 方法
- 新增 `assert self._output is not None`

### 2.16 `utils/sampling.py`

- **删除** `k = min(k, probs.shape[-1])` 保护行（k > vocab_size 时会直接报错）

### 2.17 `client_utils.py`

- **删除** `log()` 函数（6 行）

### 2.18 `server.py`（personaplex-only 文件）

这是 personaplex 新增的 WebSocket 推理服务器，相比 Moshi 原版 server.py 做了大量重构：
- 新增 NVIDIA MIT License
- 新增 `other_mimi`（第二个 Mimi 实例）
- 新增 voice prompt 下载和管理（`_get_voice_prompt_dir()`）
- 新增 `wrap_with_system_tags()` 文本处理
- 异步任务拆分（recv_loop / opus_loop / send_loop）
- 新增 `is_alive()` 心跳检测
- 自定义日志系统（`ColorizedLog`）
- CLI: 移除 `--cfg-coef`, `--lora-weight`, 新增 `--cpu-offload`, `--voice-prompt-dir`

---

## 三、PersonaPlex 新增的独立文件

| 文件 | 用途 |
|---|---|
| `personaplex/server.py` | WebSocket 推理服务器（替代 moshi.server） |
| `personaplex/offline.py` | 离线推理入口——WAV 输入 → WAV 输出 |
| `personaplex/utils/connection.py` | SSL 管理（自动下载 mkcert 生成自签名证书） |
| `personaplex/utils/logging.py` | 结构化日志（`setup_logger`, `ColorizedLog`） |

---

## 四、PersonaPlex 完全删除的 Moshi 文件

| 文件 | 原因 |
|---|---|
| `conditioners/` 整个目录 | 移除 CFG 条件注入机制 |
| `models/lm_utils.py` | 内容内联到 `lm.py` |
| `models/tts.py` | TTS 功能，PersonaPlex 不支持 |
| `modules/lora.py` | LoRA 适配器，PersonaPlex 推理不需要（微调时需要单独处理） |
| `modules/conv_test.py` | 测试代码 |
| `modules/seanet_test.py` | 测试代码 |
| `client.py` | Moshi 标准客户端 |
| `client_gradio.py` | Gradio 客户端 |
| `run_inference.py` | Moshi 推理脚本 |
| `run_tts.py` | TTS 推理脚本 |
| `utils/quantize.py` | 量化工具 |
| `utils/utils.py` | Moshi 通用工具 |

---

## 五、Streaming 架构对比

| 特性 | Moshi 原版 | PersonaPlex |
|---|---|---|
| State 基类 | `State(batch_size, device)` 带 `exec_mask`, `set_exec_mask` | `Resetable` Protocol（只有 `reset()`） |
| Streaming 卷积 | `StreamingConv1d`（基于 `previous` + `first` 重叠缓存） | `StreamingConv1d`（基于 padding + `RawStreamingConv1d`） |
| Streaming 转置卷积 | `StreamingConvTranspose1d`（基于 `partial` 累加） | `StreamingConvTranspose1d`（基于 `RawStreamingConvTranspose1d` overlap-add） |
| exec_mask | 支持（不同步推理） | **删除** |
| State 序列化 | 不支持 | 新增 `save_streaming_state()` + `load_streaming_state()` |
| 流式加法 | 无 | 新增 `StreamingAdd` |

---

## 六、量化层差异

| 特性 | Moshi 原版 | PersonaPlex |
|---|---|---|
| K-means 初始化 | 支持（`_run_kmeans`, `_init_embedding`） | **删除**（码本仅从预训练权重加载） |
| 过期码字替换 | 支持（`_check_expired_codes`） | **删除** |
| EMA 跟踪 | 支持（`cluster_usage`, `embedding_sum`） | **删除** |
| STE commit loss | 支持 | **删除** |
| q_dropout | 支持（训练时随机减少码本数） | **删除**（forward 不再应用） |
| no_quantization | 随机 mask | 新增 `no_quantization_mode` 三种模式 |
