# CS336 Assignment 1: Transformer & Optimization Project State

这份文件旨在为您在服务器上继续开发或重新启动训练提供完整的上下文同步。

## 1. 当前项目进度 (Progress Summary)
我们已经完成了 CS336 作业一的所有核心组件实现，且通过了所有的 `pytest` 单元测试。

### 已实现模块 (cs336_basics/)
- **Tokenizer**: 实现了字节级 BPE Tokenizer (`tokenizer.py`)。
- **Transformer 架构**: 
  - `RMSNorm`: 层归一化实现。
  - `RoPE (Rotary Positional Embeddings)`: 旋转位置编码。
  - `MultiHeadAttention`: 支持因果掩码 (Causal Mask) 和 RoPE。
  - `FeedForward`: 使用 SwiGLU 激活函数。
  - `TransformerBlock` & `TransformerLM`: 完整的解码器架构。
- **优化与调度**:
  - `optimizer.py`: 手动实现了 **AdamW**（解耦权重衰减、偏置修正）。
  - `schedule.py`: 实现了带有线性预热的 **Cosine 学习率调度**。
- **工具与实验**:
  - `nn_utils.py`: Softmax, CrossEntropyLoss, ClipGradNorm。
  - `checkpoint.py`: 手动实现基于 `pickle` 的模型和优化器状态保存/加载。
  - `train_tinystories_lm.py`: 用于在 TinyStories 数据集上训练的完整脚本。

## 2. 环境要求 (Environment)
- **Python**: 3.10+
- **关键库**: `torch`, `numpy`, `tqdm`, `matplotlib` (可选，用于绘图)。
- **数据集**: TinyStories (需在服务器上解压并将路径指向脚本参数)。

## 3. 如何在服务器上运行

### A. 验证安装
在 `assignment1-basics` 目录下运行：
```bash
pytest
```
确认 40+ 个测试全部通过。

### B. 启动训练 (TinyStories)
```bash
python cs336_basics/train_tinystories_lm.py \
  --input_file /path/to/tinystories.txt \
  --work_dir ./exp_v1 \
  --batch_size 16 \
  --max_steps 10000
```
*注意：请根据服务器显存调整 `batch_size`。*

### C. 实时监控
脚本会在 `--work_dir` 下自动生成：
- `log.txt`: 包含每步的 Loss 和 LR。
- `model_latest.pth`: 最新的模型权重。
- `optimizer_latest.pth`: 优化器状态。
- `loss_curve.png`: 实时生成的损失函数图表（如果配置了绘图）。

## 4. 技术细节笔记 (Technical Notes)
1. **RoPE 实现**: 计算公式使用了 $\text{cos}(\theta)$ 和 $\sin(\theta)$ 的旋转矩阵。注意频率 $\theta$ 的计算底数为 10000。
2. **AdamW 实现**: 严格遵循了 $\theta_{t} \leftarrow \theta_{t-1} - \eta_t (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})$ 的逻辑。
3. **CrossEntropy**: 为了数值稳定性，手动计算了 LogSumExp。


