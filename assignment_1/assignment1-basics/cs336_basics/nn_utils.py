from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    对输入张量在指定维度上执行 softmax。

    数值稳定实现：先减去该维度最大值，再做 exp 和归一化，
    避免 logits 很大时出现数值上溢（overflow）。

    参数：
    - in_features: 任意形状输入张量。
    - dim: 要做 softmax 的维度。

    返回：
    - 与输入同形状的概率张量，该维度上元素和为 1。
    """
    # 为了数值稳定，先减去该维度最大值（不改变 softmax 结果）
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    # 对平移后的值做指数
    exp_values = torch.exp(shifted)
    # 归一化：exp(x_i) / sum_j exp(x_j)
    return exp_values / torch.sum(exp_values, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    计算分类任务的平均交叉熵损失。

    约定：
    - inputs: 形状通常为 (batch_size, vocab_size)，表示未归一化 logits。
    - targets: 形状通常为 (batch_size,)，每个元素是正确类别索引。

    实现思路：
    1) 先对 logits 做 log_softmax（数值稳定）；
    2) 按 target 索引取出每个样本对应的对数概率；
    3) 取负后再对 batch 求平均。
    """
    # log_softmax 比先 softmax 再 log 更稳定
    log_probs = torch.log_softmax(inputs, dim=-1)
    # 取出每个样本“正确类别”的 log 概率，并取负得到 NLL（负对数似然）
    nll = -log_probs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    # 返回 batch 平均损失
    return nll.mean()


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    从一维 token 数据集中随机采样语言模型训练批次。

    给定 dataset（如 [t0, t1, t2, ...]），采样多个起点 s：
    - x = dataset[s : s + context_length]
    - y = dataset[s + 1 : s + context_length + 1]

    因此 y 始终是 x 的“右移一位标签”。

    返回：
    - x, y: 形状都为 (batch_size, context_length) 的 LongTensor。
    """
    # 起始下标的上界（不含），确保切片长度足够
    max_start = len(dataset) - context_length
    # 随机采样 batch_size 个起点（均匀采样）
    starts = torch.randint(0, max_start, (batch_size,))
    # 将 numpy 数组转成 long tensor（token id 通常用整型）
    data_tensor = torch.as_tensor(dataset, dtype=torch.long)
    # 采样输入序列 x
    x = torch.stack([data_tensor[s : s + context_length] for s in starts])
    # 采样对应标签序列 y（相对 x 右移 1 位）
    y = torch.stack([data_tensor[s + 1 : s + context_length + 1] for s in starts])
    # 放到目标设备（cpu/cuda/mps）
    return x.to(device), y.to(device)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    对参数集合的梯度执行全局 L2 范数裁剪（in-place）。

    流程：
    1) 仅收集有梯度的参数（p.grad is not None）；
    2) 计算全局梯度范数 total_norm；
    3) 若 total_norm > max_l2_norm，则按统一比例缩放所有梯度。

    这样可以抑制梯度爆炸，提升训练稳定性。
    """
    # 仅保留有梯度的参数（冻结参数或未参与反传的参数会被跳过）
    trainable_parameters = [p for p in parameters if p.grad is not None]
    # 没有可裁剪梯度时直接返回
    if not trainable_parameters:
        return

    # 计算全局 L2 范数：sqrt(sum_i ||g_i||_2^2)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in trainable_parameters]), p=2)
    # 如果已经不超过阈值，则无需裁剪
    if total_norm <= max_l2_norm:
        return

    # 计算统一缩放因子（加 1e-6 避免极小范数导致除零风险）
    scale = max_l2_norm / (total_norm + 1e-6)
    # 原地缩放每个参数的梯度
    for param in trainable_parameters:
        param.grad.mul_(scale)