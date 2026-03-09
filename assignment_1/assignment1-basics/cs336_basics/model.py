"""
Transformer 模型的核心组件实现。

这个模块包含实现 Transformer 架构所需的所有基础组件，
包括线性层、嵌入层、注意力机制、激活函数、归一化等。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
import math
import torch.nn.functional as F
import math


class Linear(nn.Module):
    """
    线性层（全连接层）的实现。
    
    功能：对输入进行线性变换 output = input @ weight.T
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        初始化线性层。
        
        Args:
            d_in (int): 输入特征维度
            d_out (int): 输出特征维度
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (*, d_in)
            
        Returns:
            输出张量，形状为 (*, d_out)
        """
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    嵌入层的实现。
    
    功能：将离散的索引（如词索引）映射到连续的向量表示。
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化嵌入层。
        
        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 嵌入维度
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) / math.sqrt(d_model))
    
    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """
        前向传播。
        
        Args:
            x: 输入张量，包含整数索引，形状为 (*)
            
        Returns:
            嵌入张量，形状为 (*, d_model)
        """
        return self.weight[x]


class SiLU(nn.Module):
    """
    SiLU (Swish) 激活函数的实现。
    
    公式：SiLU(x) = x * sigmoid(x)
    """
    
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (*)
            
        Returns:
            激活后的张量，形状为 (*)
        """
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) 的实现。
    
    这是一个简化的归一化层，基于均方根而不是标准差，
    在 Transformer 模型（如 LLaMA）中广泛使用。
    
    公式：RMSNorm(x) = x / RMS(x) * weight
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6, device=None):
        """
        初始化 RMSNorm 层。
        
        Args:
            d_model (int): 模型维度
            eps (float): 数值稳定性的小常数，默认 1e-6
            device: 设备参数（可选）
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        前向传播。
        
        Args:
            x: 输入张量，最后一个维度为 d_model
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算 RMS (Root Mean Square)
        # 保持最后一个维度，对其他维度求平方的均值
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并乘以权重
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络的实现。
    
    SwiGLU 是 GLU (Gated Linear Unit) 的一种变体，使用 SiLU 作为门函数。
    结构：
    - W1 和 W3 将输入从 d_model 投影到 d_ff
    - 应用 SiLU 激活到 W1 的输出
    - 将 W1 的激活输出和 W3 的输出逐元素相乘（门机制）
    - W2 将结果投影回 d_model
    
    公式：SwiGLU(x) = (W1(x) * SiLU(W1(x))) @ W2
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        初始化 SwiGLU 层。
        
        Args:
            d_model (int): 输入输出维度
            d_ff (int): 内部隐藏层维度
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.silu = SiLU()
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        前向传播。
        
        Args:
            x: 输入张量，最后一个维度为 d_model
            
        Returns:
            输出张量，形状与输入相同
        """
        # W1(x) 和 W3(x) 的输出
        gate = self.w1(x)
        value = self.w3(x)
        
        # 应用 SiLU 激活到第一个输出，然后与第二个输出相乘
        gated = self.silu(gate) * value
        
        # 通过 W2 投影回 d_model
        return self.w2(gated)


class RoPE(nn.Module):
    """
    RoPE (Rotary Position Embeddings) 的实现。
    
    RoPE 通过旋转矩阵对查询和键进行位置编码，
    是现代 LLM（如 LLaMA）中的重要组件。
    
    核心思想：位置信息通过旋转 Q 和 K 来编码。
    对于位置 m 和维度对 (2i, 2i+1)，旋转角为 m * theta^(-2i/d)。
    """
    
    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048):
        """
        初始化 RoPE 层。
        
        Args:
            d_k (int): 查询/键的维度（通常是 d_model / num_heads）
            theta (float): RoPE 的底数，默认 10000
            max_seq_len (int): 最大序列长度，用于预计算旋转矩阵
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # 预计算旋转频率
        # freqs[i] = theta^(-2i/d) for i in 0..d/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        # 注册为缓冲区而不是参数，这样它不会被加入状态字典
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"],
        positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        应用 RoPE 到输入张量。
        
        Args:
            x: 查询或键张量，形状为 (*, seq_len, d_k)
            positions: 可选的位置张量，如果为 None 则使用连续位置 [0, 1, ..., seq_len-1]
            
        Returns:
            旋转后的张量，形状与输入相同
        """
        seq_len = x.shape[-2]
        device = x.device
        dtype = x.dtype
        
        # 如果没有提供位置，使用默认的连续位置
        if positions is None:
            positions = torch.arange(seq_len, device=device, dtype=dtype)
        else:
            positions = positions.float()
        
        # 计算 m * theta^(-2i/d)
        # positions shape: (..., seq_len)
        # inv_freq shape: (d_k//2,)
        # freqs shape: (..., seq_len, d_k//2)
        freqs = positions.unsqueeze(-1) * self.inv_freq
        
        # 将 x 转换为复数形式，其中相邻的两个维度构成复数的实部和虚部
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        # 创建旋转因子 e^(i*freqs)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        # 应用旋转：x_rotated = x * e^(i*freqs)
        x_rotated = x_complex * freqs_complex
        
        # 转换回实数形式
        x_rotated = torch.view_as_real(x_rotated).flatten(-2)
        
        return x_rotated


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力的实现。
    
    这是 Transformer 中的核心注意力机制。
    公式：Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    
    其中：
    - Q: 查询矩阵
    - K: 键矩阵
    - V: 值矩阵
    - d_k: 键/查询的维度
    """
    
    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        """
        前向传播。
        
        Args:
            Q: 查询张量，形状为 (*, queries, d_k)
            K: 键张量，形状为 (*, keys, d_k)
            V: 值张量，形状为 (*, values, d_v)
            mask: 可选的掩码张量，形状为 (*, queries, keys)，
                  True 表示要屏蔽的位置（将被替换为 -inf）
            
        Returns:
            注意力输出张量，形状为 (*, queries, d_v)
        """
        d_k = Q.shape[-1]
        
        # 计算注意力分数：Q @ K.T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 将掩码为 False 的位置替换为 -inf（注意：这里的语义是 True=保留，False=屏蔽）
            scores = scores.masked_fill(~mask, -1e9)
        
        # 应用 softmax 获得注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 处理可能出现的 NaN（当所有分数都是 -inf 时）
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # 计算输出：attn_weights @ V
        output = torch.matmul(attn_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    CausalMultiHeadAttention 是因果多头注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    每个头的公式都是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        wq: (d_model, d_k) 查询的权重
        wk: (d_model, d_k) 键的权重
        wv: (d_model, d_v) 值的权重
        wo: (d_model, d_model) 输出的权重
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # mask 语义：True=保留，False=屏蔽
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)
        
    def forward(self, x, wq, wk, wv, wo, mask=None)->torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #现在的形状是(batch_size, n_heads, seq_len, head_dim)
        # 创建mask，用于防止当前位置的token看到未来的token。 (Re-enabled Causal Mask)
        if mask is None:
             # 因果 mask：保留过去和当前（下三角+对角线），屏蔽未来
             mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
             mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T
        return out


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    带 RoPE 的多头注意力实现。
    
    在标准多头注意力的基础上添加 RoPE 位置编码。
    """
    
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 2048):
        """
        初始化带 RoPE 的多头注意力层。
        
        Args:
            d_model (int): 模型维度
            num_heads (int): 注意力头数
            theta (float): RoPE 的底数
            max_seq_len (int): 最大序列长度
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE 和注意力
        self.rope = RoPE(self.d_k, theta, max_seq_len)
        self.attention = ScaledDotProductAttention()
    
    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        positions: Int[Tensor, "batch seq_len"] | None = None,
        mask: Bool[Tensor, "batch seq_len seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
            positions: 可选的位置张量
            mask: 可选的掩码张量
            
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 转置: (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 应用 RoPE 到 Q 和 K
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # 调整 positions 形状以匹配 Q 和 K 的头维度: (batch, seq_len) -> (batch, 1, seq_len)
        positions_for_rope = positions.unsqueeze(1)
            
        # 应用 RoPE
        Q = self.rope(Q, positions_for_rope)
        K = self.rope(K, positions_for_rope)
        
        # 扩展掩码
        if mask is not None:
            mask = mask.unsqueeze(1)
        else:
            # 默认为因果掩码：保留过去和当前，屏蔽未来
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 应用注意力
        attn_output = self.attention(Q, K, V, mask)
        
        # 拼接多个头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.output_proj(attn_output)
        
        return output


class CausalMultiHeadAttentionWithRoPE(nn.Module):
    """
    带 RoPE 的因果多头注意力实现（接受外部权重）。
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        
        # RoPE 和注意力
        self.rope = RoPE(self.head_dim, theta, max_seq_len)
        self.attention_fn = ScaledDotProductAttention()
    
    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # mask 语义：True=保留，False=屏蔽
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)
    
    def forward(self, x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 投影到 Q, K, V
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T
        
        # 分割成多个头并转置
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE
        if token_positions.dim() == 1:
            token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        positions_for_rope = token_positions.unsqueeze(1)  # (batch, 1, seq_len)
        
        q = self.rope(q, positions_for_rope)
        k = self.rope(k, positions_for_rope)
        
        # 创建因果 mask：保留过去和当前，屏蔽未来
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # 应用注意力
        out = self.attention(q, k, v, mask)
        
        # 拼接多个头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        out = out @ wo.T
        return out


class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        attn_q_proj_weight (torch.Tensor): 查询的权重
    """
    def __init__(self, d_model:int, n_heads:int, d_ff:int, max_seq_len:int, theta:float, attn_q_proj_weight:torch.Tensor, attn_k_proj_weight:torch.Tensor, attn_v_proj_weight:torch.Tensor, attn_o_proj_weight:torch.Tensor, ln1_weight:torch.Tensor, ln2_weight:torch.Tensor, ffn_w1_weight:torch.Tensor, ffn_w2_weight:torch.Tensor, ffn_w3_weight:torch.Tensor, device=None):
        super(TransformerBlock, self).__init__()
        # 权重
        self.attn_q_proj_weight = attn_q_proj_weight
        self.attn_k_proj_weight = attn_k_proj_weight
        self.attn_v_proj_weight = attn_v_proj_weight
        self.attn_o_proj_weight = attn_o_proj_weight

        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.ffn_w1_weight = ffn_w1_weight
        self.ffn_w2_weight = ffn_w2_weight
        self.ffn_w3_weight = ffn_w3_weight
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        
        # 初始化子模块
        self.rms_norm1 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm1.load_state_dict({"weight": self.ln1_weight})
        
        self.rms_norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm2.load_state_dict({"weight": self.ln2_weight})
        
        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({"w1.weight": self.ffn_w1_weight, "w2.weight": self.ffn_w2_weight, "w3.weight": self.ffn_w3_weight})
        
        self.causal_multi_head_attention = CausalMultiHeadAttentionWithRoPE(d_model, n_heads, max_seq_len, theta, device)

    def forward(self, in_features:torch.Tensor):
        token_positions = torch.arange(in_features.shape[1], device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1, self.attn_q_proj_weight, self.attn_k_proj_weight, self.attn_v_proj_weight, self.attn_o_proj_weight, token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out


class TransformerLanguageModel(nn.Module):
    """
    Transformer 语言模型的实现。
    
    这是一个完整的自回归语言模型，包含：
    1. 词嵌入层
    2. 多个 Transformer 块
    3. 最后的 RMSNorm
    4. 语言模型输出头
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        初始化 Transformer 语言模型。
        
        Args:
            vocab_size (int): 词汇表大小
            context_length (int): 上下文长度（最大序列长度）
            d_model (int): 模型维度
            num_layers (int): Transformer 块的数量
            num_heads (int): 注意力头数
            d_ff (int): 前馈网络的隐藏维度
            rope_theta (float): RoPE 参数
            eps (float): RMSNorm 的数值稳定性参数
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.eps = eps
        self._loaded_weights: dict[str, Tensor] | None = None
        
        # 词嵌入
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # 最后的归一化
        self.ln_final = RMSNorm(d_model, eps)
        
        # 语言模型输出头
        self.lm_head = Linear(d_model, vocab_size)

    def load_state_dict(self, state_dict, strict: bool = True):
        self._loaded_weights = state_dict

        self.token_embeddings.load_state_dict(
            {"weight": state_dict["token_embeddings.weight"]}
        )
        self.ln_final.load_state_dict({"weight": state_dict["ln_final.weight"]})
        self.lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})

        return nn.modules.module._IncompatibleKeys([], [])
    
    def forward(
        self,
        token_ids: Int[Tensor, "batch seq_len"],
        positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        """
        前向传播。
        
        Args:
            token_ids: 输入的词索引，形状为 (batch, seq_len)
            positions: 可选的位置张量
            
        Returns:
            预测的词概率分布，形状为 (batch, seq_len, vocab_size)
        """
        _, seq_len = token_ids.shape
        
        # 获取词嵌入
        x = self.token_embeddings(token_ids)
        
        if self._loaded_weights is None:
            raise RuntimeError("TransformerLanguageModel must load weights before forward.")

        # 通过每个 Transformer 块（按层从权重字典构造）
        for layer_idx in range(self.num_layers):
            block = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                attn_q_proj_weight=self._loaded_weights[f"layers.{layer_idx}.attn.q_proj.weight"],
                attn_k_proj_weight=self._loaded_weights[f"layers.{layer_idx}.attn.k_proj.weight"],
                attn_v_proj_weight=self._loaded_weights[f"layers.{layer_idx}.attn.v_proj.weight"],
                attn_o_proj_weight=self._loaded_weights[f"layers.{layer_idx}.attn.output_proj.weight"],
                ln1_weight=self._loaded_weights[f"layers.{layer_idx}.ln1.weight"],
                ln2_weight=self._loaded_weights[f"layers.{layer_idx}.ln2.weight"],
                ffn_w1_weight=self._loaded_weights[f"layers.{layer_idx}.ffn.w1.weight"],
                ffn_w2_weight=self._loaded_weights[f"layers.{layer_idx}.ffn.w2.weight"],
                ffn_w3_weight=self._loaded_weights[f"layers.{layer_idx}.ffn.w3.weight"],
                device=token_ids.device,
            )
            x = block(x)
        
        # 最后的归一化
        x = self.ln_final(x)
        
        # 通过输出头获得 logits
        logits = self.lm_head(x)
        
        return logits
