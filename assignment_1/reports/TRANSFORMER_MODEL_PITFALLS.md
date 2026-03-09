# CS336 Assignment 1（Transformer 部分）踩坑记录

## 1. 目标与范围

本记录聚焦 Assignment 1 中模型部分（`test_model.py`）从失败到全通过的调试过程，覆盖：

- Multi-Head Self-Attention（含 RoPE）
- TransformerBlock
- TransformerLanguageModel（含截断输入）

最终状态：`tests/test_model.py` 全部 13 项测试通过。

---

## 2. 核心踩坑清单

## 坑 1：Attention 缩放因子写错（`d_model` vs `head_dim`）

### 现象

- `test_multihead_self_attention` 数值不匹配快照。
- 输出整体统计量（均值/方差）偏离参考实现。

### 根因

缩放点积注意力应使用：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

其中 $d_k$ 是每个头的维度（`head_dim = d_model / n_heads`），而不是 `d_model`。

### 修复

- 将缩放项改为 `sqrt(head_dim)`。

### 验证

- `test_multihead_self_attention` 通过。

---

## 坑 2：因果 Mask 方向反了（保留未来、屏蔽过去）

### 现象

- 注意力行为看起来“能偷看未来”，导致快照误差明显。

### 根因

mask 语义与实现不一致：

- 代码里的 attention 逻辑是 `True=保留, False=屏蔽`。
- 但上游构造 mask 时方向或比较条件写反，等价于上三角保留。

### 修复

- 统一语义：`True=保留，False=屏蔽`。
- 使用标准因果下三角：`torch.tril(...)`。

### 验证

- `test_scaled_dot_product_attention`
- `test_multihead_self_attention`
- `test_multihead_self_attention_with_rope`

均通过。

---

## 坑 3：`TransformerBlock` 与 `TransformerLanguageModel` 调用模式冲突

### 现象

- `test_transformer_block` 能通过；
- 但 `test_transformer_lm` / `test_transformer_lm_truncated_input` 失败。

### 根因

两套调用范式不一致：

1. `run_transformer_block`：显式把每层权重传给 `TransformerBlock(...)`。
2. `run_transformer_lm`：先构造 `TransformerLanguageModel(...)`，再 `load_state_dict(weights)`。

如果 `TransformerLanguageModel` 内部直接构造一组“随机参数化 block”，并假设常规 state_dict 逐层灌入，会与当前 `TransformerBlock` 的显式权重注入设计冲突。

### 修复

- `TransformerLanguageModel` 改为“权重驱动前向”：
  - 在 `load_state_dict` 中缓存 `weights`；
  - `forward` 时按层从缓存字典取权重，构造/调用 `TransformerBlock`；
  - `token_embeddings`、`ln_final`、`lm_head` 从同一份权重中加载。

### 验证

- `test_transformer_lm` 通过。
- `test_transformer_lm_truncated_input` 通过。

---

## 坑 4：`RMSNorm eps` 与快照精度的敏感性

### 现象

- 有时仅剩微小误差（数量级 `1e-5 ~ 1e-4`），但快照阈值仍可能失败。

### 根因

`eps` 从 `1e-6` 改到 `1e-5` 会引入稳定但可观测的数值差异，尤其在多层残差叠加后被放大。

### 修复

- 与测试/参考配置保持一致：模型路径中统一使用 `eps=1e-5`。

### 验证

- `test_rmsnorm` 与下游模型快照测试同时通过。

---

## 坑 5：`load_state_dict` 语义与 assignment 适配器预期不一致

### 现象

- 直接依赖默认 `load_state_dict` 时，某些模块 key 对不上或行为不符合 adapter 假设。

### 根因

assignment 的 adapter 更接近“功能验证驱动”而非“严格模块参数同构”；需要保证 adapter 调用路径中的行为正确，而不是强行复刻标准层级化参数树。

### 修复

- 在 `TransformerLanguageModel` 内定制 `load_state_dict`，仅加载必需权重并缓存完整权重字典供 block 前向使用。

### 验证

- `run_transformer_lm -> lm.load_state_dict(weights) -> lm(in_indices)` 路径通过全部相关测试。

---

## 3. 调试策略复盘（有效做法）

- 先收敛局部：先保 `attention` 测试过，再进 `block`，最后处理 `lm`。
- 发现“单测通过但集成失败”时，优先检查调用约定与接口签名。
- 所有数值相关问题优先核对三项：缩放因子、mask 语义、norm eps。
- 用最小验证集反复跑（单测点跑），最终再跑全量回归。

---

## 4. 最终验证结果

已执行：

- `pytest tests/test_model.py -q`

结果：

- `13 passed`

说明本阶段（Transformer 模型部分）实现与修复已经稳定完成。
