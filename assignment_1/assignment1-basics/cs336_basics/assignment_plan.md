# CS336 Assignment 1 Plan

## 阶段 0：环境与资料准备
- 通读 `cs336_spring2025_assignment1_basics.pdf`，列出所有书面题答案需求。
- 安装 `uv` 并执行 `uv run pytest`，确认当前均报 `NotImplementedError`。
- 依 README 下载 TinyStories 与 OWT 数据，准备训练/验证集。

## 阶段 1：Tokenizer 与 BPE 训练
1. **实现 tokenizer 接口**  
   - 在 `cs336_basics/tokenizer.py`（待创建）中完成 UTF-8 字节级 BPE tokenizer，支持 `encode`, `decode`, `encode_iterable`，以及 special tokens。
2. **实现 BPE 训练**  
   - 在 `cs336_basics/bpe_training.py` 实现高效 pair 统计与合并，满足 `test_train_bpe_speed`。
3. **连接 adapters**  
   - 填写 `tests/adapters.py` 中的 `get_tokenizer()`、`run_train_bpe()`。
4. **验证**  
   - 运行 `uv run pytest tests/test_tokenizer.py tests/test_train_bpe.py tests/test_data.py`。

## 阶段 2：基础 NN 工具
1. 在 `cs336_basics/nn_utils.py` 实现：`softmax`（数值稳定）、`cross_entropy`、`get_batch`、`gradient_clipping`。
2. 在 adapters 中补齐 `run_softmax`, `run_cross_entropy`, `run_get_batch`, `run_gradient_clipping`。
3. 运行 `tests/test_nn_utils.py` 与 `tests/test_data.py`。

## 阶段 3：Transformer 组件
1. 模块拆分建议
   - `cs336_basics/modules.py`: `Linear`, `Embedding`, `SwiGLU`, `RMSNorm`, `RoPE`, `ScaledDotProductAttention`, `MultiheadAttention`, `TransformerBlock`, `TransformerLM`。
2. 确保权重 shape 与 snapshot state dict 对齐，支持 RoPE、可变长度、mask。
3. 在 adapters 中完成对应 `run_*` glue。
4. 运行 `tests/test_model.py`。

## 阶段 4：优化器与调度
1. `cs336_basics/optimizer.py`: 实现继承 `torch.optim.Optimizer` 的 AdamW（含 bias correction、decoupled weight decay）。
2. `cs336_basics/schedule.py`: 实现带线性 warmup 的余弦学习率调度函数，供 `run_get_lr_cosine_schedule`。
3. 适配 adapters：`get_adamw_cls()`, `run_get_lr_cosine_schedule()`。
4. 运行 `tests/test_optimizer.py`。

## 阶段 5：序列化与训练循环
1. `cs336_basics/checkpoint.py`: 实现 `save_checkpoint` / `load_checkpoint`（模型、优化器、迭代数）。
2. 训练脚本：
   - 组合 tokenizer、数据加载、模型构建、批采样、loss/backprop、梯度裁剪、调度、checkpoint。
   - 支持 CPU/MPS/GPU，包含日志与采样生成。
3. 完成 adapters：`run_save_checkpoint()`, `run_load_checkpoint()`。
4. 全量 `uv run pytest` 验证。

## 阶段 6：实验与写作
- 使用 TinyStories 完成端到端训练，记录 perplexity 与生成样本。
- 按 PDF 回答书面题，撰写 `writeup.pdf`（建议 LaTeX）。
- 若上榜需求，按 leaderboard README 提交 OWT 结果。

## 迭代策略
- 每完成一阶段立即运行对应测试，保证小步快测。
- adapters 仅做“粘合”调用，核心实现放在 `cs336_basics` 模块内。
- 优先确保 CPU 版本正确，再考虑 GPU/MPS 优化。
- 注意数值稳定性（logits normalization、RMSNorm eps、AdamW fp32）。
