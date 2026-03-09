# CS336 Assignment 1 总项目完成笔记（长期回顾版）

## 1. 这份笔记的用途

这份文档用于在很久之后快速恢复对 Assignment 1 全流程的记忆：

- 理论主线：Tokenizer -> Transformer LM -> Loss/Optimizer -> Training Loop -> Decoding -> Experiments
- 代码主线：`cs336_basics/` 中各模块职责
- 公式主线：作业 PDF（`cs336_spring2025_assignment1_basics.pdf`）中的核心数学定义
- 工程主线：测试驱动实现与常见坑位

当前仓库状态：`pytest -q` 已通过（`46 passed, 2 skipped`）。

---

## 2. 项目总览（按 PDF 结构）

Assignment 1 的核心任务可以压缩为 4 条线：

1. Tokenization 线：字节级 BPE 的训练、编码、解码。
2. Model 线：实现 decoder-only Transformer（含 RoPE、RMSNorm、SwiGLU）。
3. Training 线：softmax/cross-entropy、AdamW、学习率调度、梯度裁剪、checkpoint。
4. Experiment 线：在 TinyStories/OWT 上训练、生成、做消融分析。

---

## 3. 关键数学公式速查（来自 handout）

## 3.1 Softmax 与数值稳定

PDF 公式（Eq. 10）：

$$
\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_j \exp(v_j)}
$$

工程实现关键点：

- 用 `v - max(v)` 再做 `exp`，避免上溢。

## 3.2 Scaled Dot-Product Attention

PDF 公式（Eq. 11）：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right)V
$$

工程实现关键点：

- 缩放维度必须是 `d_k`（每头维度），不是 `d_model`。
- 因果 mask 语义要统一：`True=保留，False=屏蔽`。

## 3.3 Multi-Head 与 Self-Attention

PDF 公式（Eq. 12-14）：

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_1,\ldots,head_h)
$$

$$
head_i = \mathrm{Attention}(Q_i,K_i,V_i)
$$

$$
\mathrm{MHSA}(x)=W_O\,\mathrm{MultiHead}(W_Qx, W_Kx, W_Vx)
$$

工程实现关键点：

- 张量形状常用：`(B,T,D) -> (B,H,T,d_k) -> attention -> (B,T,D)`。

## 3.4 RMSNorm

PDF 公式（Eq. 4）：

$$
\mathrm{RMSNorm}(a_i)=\frac{a_i}{\mathrm{RMS}(a)}g_i,
\quad \mathrm{RMS}(a)=\sqrt{\frac{1}{d_{model}}\sum_i a_i^2 + \epsilon}
$$

工程实现关键点：

- 常用 `eps=1e-5`；不同 `eps` 可能造成快照误差累积。

## 3.5 SwiGLU / FFN

PDF 公式（Eq. 5-7）：

$$
\mathrm{SiLU}(x)=x\sigma(x)
$$

$$
\mathrm{SwiGLU}(x,W_1,W_2,W_3)=W_2\big(\mathrm{SiLU}(W_1x)\odot W_3x\big)
$$

## 3.6 语言模型目标（Cross-Entropy）

PDF 公式（Eq. 16-17）：

$$
\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^m -\log p_\theta(x_{i+1}|x_{1:i})
$$

$$
p(x_{i+1}|x_{1:i}) = \mathrm{softmax}(o_i)[x_{i+1}]
$$

工程实现关键点：

- `log_softmax + gather(index by targets)` 是最稳妥实现。

## 3.7 Perplexity

PDF 公式（Eq. 18）：

$$
\mathrm{PPL}=\exp\left(\frac{1}{m}\sum_{i=1}^m \ell_i\right)
$$

## 3.8 SGD 与 AdamW

PDF 给出 SGD（Eq. 19）与变体（Eq. 20）；作业要求实现 AdamW（decoupled weight decay）。

AdamW 核心更新（简化写法）：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$$
\theta \leftarrow \theta - \alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\quad
\text{and}
\quad
\theta \leftarrow \theta - \alpha\lambda\theta
$$

## 3.9 学习率调度（warmup + cosine）

PDF 规则：

- `t < Tw`：线性 warmup
- `Tw <= t <= Tc`：余弦退火
- `t > Tc`：固定在 `alpha_min`

## 3.10 解码（采样）

PDF 给出 temperature 与 top-p（nucleus）采样：

$$
\mathrm{softmax}(v,\tau)_i = \frac{\exp(v_i/\tau)}{\sum_j \exp(v_j/\tau)}
$$

top-p 的核心思想：保留概率累计达到 `p` 的最小词集，再重归一化采样。

---

## 4. 代码结构回忆图

本仓库关键文件（实现层）：

- `cs336_basics/tokenizer.py`
- `cs336_basics/model.py`
- `cs336_basics/nn_utils.py`
- `cs336_basics/optimizer.py`
- `cs336_basics/schedule.py`
- `cs336_basics/checkpoint.py`

测试胶水层：

- `tests/adapters.py` 只负责把测试输入转发到 `cs336_basics` 实现。

测试层：

- `tests/test_model.py`
- `tests/test_tokenizer.py`
- `tests/test_train_bpe.py`
- `tests/test_nn_utils.py`
- `tests/test_optimizer.py`
- `tests/test_serialization.py`

---

## 5. 我这次实现/调试的关键工程结论

## 5.1 注意力模块

- `sqrt(d_k)` 是最常见错误点。
- causal mask 的方向错误会直接导致快照偏移。
- RoPE 只作用于 Q/K，且按 head 维独立应用。

## 5.2 TransformerBlock 与 LM 适配

- block 级测试和 lm 级测试可能有不同权重注入路径。
- 需要保证 `adapters` 只是 glue，核心逻辑在 `cs336_basics`。

## 5.3 Tokenizer 解码边界

- 对非完整 UTF-8 字节片段，解码要允许 replacement 字符：
  - `decode(..., errors='replace')`

## 5.4 Optimizer / Checkpoint 约束

- 不偷懒直接把 `torch.optim.AdamW` 当作最终实现。
- 按作业精神在 `cs336_basics` 中手写核心逻辑，再由 adapter 调用。

---

## 6. 高价值记忆点（容易忘但很关键）

1. Unicode 与 UTF-8：字节级 tokenization 不会 OOV。
2. BPE 本质：高频相邻符号对迭代合并，得到压缩更高的子词词表。
3. Pre-norm Transformer：`x + sublayer(norm(x))`，训练更稳。
4. RoPE 本质：对 Q/K 进行位置相关旋转，让注意力携带相对位置信息。
5. CE 与 PPL：PPL 是 CE 的指数形态，更直观但数值更“放大”。
6. AdamW 区别点：权重衰减与梯度更新解耦。
7. 训练系统工程：数据加载、梯度裁剪、调度、checkpoint 与模型同等重要。

---

## 7. 快速复现指令（以后回来看先跑这个）

在 `assignment1-basics` 目录：

```bash
pytest -q
```

若只查模型：

```bash
pytest tests/test_model.py -q
```

若只查优化器与序列化：

```bash
pytest tests/test_optimizer.py tests/test_serialization.py -q
```

---

## 8. 学习路径建议（过很久后重启）

建议用 30-40 分钟恢复记忆：

1. 先看本笔记第 3 节公式速查（建立理论框架）。
2. 再看第 4 节文件地图（建立代码定位能力）。
3. 然后跑第 7 节命令（建立反馈闭环）。
4. 最后看 `reports/TRANSFORMER_MODEL_PITFALLS.md`（回忆真实坑位与排障路径）。

---

## 9. 总结

Assignment 1 的真正收获不是“把测试跑绿”本身，而是把以下能力串起来：

- 从数学定义到张量实现的映射能力。
- 从局部模块正确到系统级集成正确的工程能力。
- 从单次调通到可复现、可维护、可回忆的文档化能力。

这三件事，正是后续做更大模型、更长训练、更复杂系统时的核心基础。
