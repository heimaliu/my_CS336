# CS336 Assignment 1（Transformer 模型部分）详细作业报告

## 1. 报告范围

本报告覆盖 Assignment 1 的 Transformer 模型实现与调试闭环，具体包含：

- 基础模块：`Linear`、`Embedding`、`SiLU`、`RMSNorm`、`SwiGLU`
- 注意力模块：`ScaledDotProductAttention`、`MultiHeadAttention`、`RoPE`
- 结构模块：`TransformerBlock`、`TransformerLanguageModel`
- 测试目标：`tests/test_model.py`

最终结果：模型测试全通过（13/13）。

---

## 2. 架构与实现思路

## 2.1 基础组件

- `Linear`：按 $y=xW^T$ 计算，权重形状 `(d_out, d_in)`。
- `Embedding`：按 token id 做索引映射，输出形状 `(..., d_model)`。
- `SiLU`：实现 $\text{SiLU}(x)=x\sigma(x)$。
- `RMSNorm`：

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\odot g
$$

- `SwiGLU`：使用三矩阵结构（`w1/w2/w3`），执行门控前馈。

## 2.2 注意力与位置编码

- Scaled Dot-Product Attention：

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- 多头拆分：`(B, T, D) -> (B, H, T, d_k)`，再在 head 维并行计算。
- 因果 mask：统一语义 `True=保留, False=屏蔽`，采用下三角保留历史。
- RoPE：仅作用于 `Q/K`，不作用于 `V`。

## 2.3 TransformerBlock

采用 pre-norm 结构：

1. `x1 = x + Attention(RMSNorm(x))`
2. `out = x1 + FFN(RMSNorm(x1))`

该实现同时对齐测试适配器的“显式权重注入”路径。

## 2.4 TransformerLanguageModel

本次关键设计是与测试适配器的调用约定对齐：

- 适配器路径：先构造 `TransformerLanguageModel`，再 `load_state_dict(weights)`。
- 模型内实现采用“权重驱动前向”：
  - `load_state_dict` 缓存完整权重；
  - `forward` 按层读取 `layers.{i}.*` 权重，执行 block 计算；
  - 头尾模块（`token_embeddings`、`ln_final`、`lm_head`）同步按权重加载。

该方式与 block 级适配器（显式传权重）共存，不相互破坏。

---

## 3. 关键问题与修复过程

## 3.1 Attention 缩放维度错误

- 问题：使用 `sqrt(d_model)` 导致数值偏差。
- 修复：改为 `sqrt(head_dim)`。

## 3.2 因果掩码语义混乱

- 问题：mask 方向或填充条件反向，出现“未来可见”。
- 修复：统一为 `True=保留`，并使用标准下三角掩码。

## 3.3 Block/LM 两种权重注入范式冲突

- 问题：block 级测试与 lm 级测试对模块参数管理方式不一致。
- 修复：LM 改为显式缓存并按层取权重执行前向，消除构造时参数树不匹配问题。

## 3.4 数值精度边界（`eps`）

- 问题：`RMSNorm eps` 不一致时出现小误差积累。
- 修复：对齐到 `1e-5`，通过快照阈值。

详见配套文档：`TRANSFORMER_MODEL_PITFALLS.md`。

---

## 4. 测试与验证

## 4.1 关键节点验证

- `test_multihead_self_attention`：通过
- `test_multihead_self_attention_with_rope`：通过
- `test_transformer_block`：通过
- `test_transformer_lm`：通过
- `test_transformer_lm_truncated_input`：通过

## 4.2 全量模型测试

执行命令：

- `pytest tests/test_model.py -q`

结果：

- `13 passed in 0.62s`

说明 Transformer 模型部分在当前代码状态下完成闭环验证。




