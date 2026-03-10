from __future__ import annotations

import argparse
import pickle
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from cs336_basics.model import Embedding, Linear, MultiHeadAttentionWithRoPE, RMSNorm, SwiGLU
from cs336_basics.nn_utils import cross_entropy, get_batch, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.schedule import get_lr_cosine_schedule
from cs336_basics.tokenizer import BPETokenizer, train_bpe


class TrainableTransformerBlock(nn.Module):
    """
    单个 Transformer 解码器块。
    包含：
    1. RMSNorm 归一化
    2. 带有旋转位置编码 (RoPE) 的多头自注意力机制
    3. 第二层 RMSNorm
    4. SwiGLU 前馈神经网络 (FFN)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, rope_theta: float, eps: float = 1e-5):
        super().__init__()
        # 注意力机制前的归一化
        self.ln1 = RMSNorm(d_model, eps)
        # 核心：带有 RoPE 的多头自注意力
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, rope_theta, context_length)
        # 前馈网络前的归一化
        self.ln2 = RMSNorm(d_model, eps)
        # 高性能的 SwiGLU 激活函数实现的前馈网络
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # 残差连接 1: x + Attention(Norm(x))
        x = x + self.attn(self.ln1(x), positions=positions)
        # 残差连接 2: x + FFN(Norm(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TrainableTransformerLM(nn.Module):
    """
    完整的 Transformer 语言模型 (Decoder-only 架构)。
    参数配置决定了模型的大小（深度、宽度）：
    - vocab_size: 词表大小 (e.g., 5000)
    - context_length: 最大序列长度 (e.g., 128)
    - d_model: 隐藏层维度，控制模型的“宽度” (e.g., 256)
    - num_layers: Transformer 块的数量，控制模型的“深度” (e.g., 4 层)
    - num_heads: 多头注意力的头数
    - d_ff: FFN 层的隐藏维度 (通常是 d_model 的 4 倍)
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
        super().__init__()
        self.context_length = context_length
        # 1. 词嵌入层：将 Token ID 映射为向量
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # 2. 堆叠 N 层 Transformer 块 (深度由 num_layers 决定)
        self.layers = nn.ModuleList(
            [
                TrainableTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
                for _ in range(num_layers)
            ]
        )
        
        # 3. 最终层归一化
        self.ln_final = RMSNorm(d_model, eps)
        
        # 4. 输出头：将向量映射回词表大小，用于预测下一个词
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        # 为 RoPE 生成位置信息
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        
        # 词嵌套
        x = self.token_embeddings(token_ids)
        
        # 逐层通过 Transformer 块
        for layer in self.layers:
            x = layer(x, positions)
            
        # 归一化并输出 logits
        x = self.ln_final(x)
        return self.lm_head(x)


@torch.no_grad()
def evaluate(model: nn.Module, data: np.ndarray, batch_size: int, context_length: int, device: str, eval_steps: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def generate(model: TrainableTransformerLM, tokenizer: BPETokenizer, prompt: str, device: str, max_new_tokens: int = 80, temperature: float = 1.0) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.context_length :]
        logits = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def load_text(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_chars is not None and max_chars > 0:
        return text[:max_chars]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny LM on TinyStories using custom cs336_basics modules")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--train_path", type=Path)
    parser.add_argument("--valid_path", type=Path)
    parser.add_argument("--work_dir", type=Path, default=Path("runs/tinystories_simple"))

    parser.add_argument("--max_train_chars", type=int, default=2_000_000)
    parser.add_argument("--max_valid_chars", type=int, default=400_000)
    parser.add_argument("--vocab_size", type=int, default=5000)

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--wandb_project", type=str, default="cs336_tinystories")
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    # 如果提供了 YAML 配置文件，则加载并覆盖/补充参数
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config_args = yaml.safe_load(f)
            # 将字典中的值更新到 args 对象中
            for key, value in config_args.items():
                # 对于路径类型的参数，需要特殊处理转为 Path 对象
                if key in ["train_path", "valid_path", "work_dir"] and value:
                    setattr(args, key, Path(value))
                else:
                    setattr(args, key, value)
        print(f"[info] loaded config from {args.config}")

    if getattr(args, "wandb_enabled", False) and wandb is None:
        raise ImportError("wandb is enabled but not installed. Please run: pip install wandb")

    # 检查必要参数是否已通过 YAML 或 命令行 提供
    if not args.train_path or not args.valid_path:
        parser.error("train_path and valid_path are required (via command line or config file)")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tok_cache = args.work_dir / "tokenizer.pkl"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[info] device={device}")

    if getattr(args, "wandb_enabled", False):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = args.wandb_run_name if args.wandb_run_name else f"tinystories_{timestamp}"
        wandb_config = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        }
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        print(f"[info] wandb enabled: project={args.wandb_project}, run={run_name}")

    train_text = load_text(args.train_path, args.max_train_chars)
    valid_text = load_text(args.valid_path, args.max_valid_chars)

    if tok_cache.exists():
        with open(tok_cache, "rb") as f:
            vocab, merges = pickle.load(f)
        print("[info] loaded cached tokenizer")
    else:
        tmp_train = args.work_dir / "train_subset.txt"
        tmp_train.write_text(train_text, encoding="utf-8")
        vocab, merges = train_bpe(tmp_train, args.vocab_size, ["<|endoftext|>"])
        with open(tok_cache, "wb") as f:
            pickle.dump((vocab, merges), f)
        print("[info] tokenizer trained and cached")

    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])

    train_ids = np.array(tokenizer.encode(train_text), dtype=np.int64)
    valid_ids = np.array(tokenizer.encode(valid_text), dtype=np.int64)
    print(f"[info] train tokens={len(train_ids)}, valid tokens={len(valid_ids)}")

    model = TrainableTransformerLM(
        vocab_size=len(vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    model.train()
    for step in range(1, args.max_steps + 1):
        lr = get_lr_cosine_schedule(step - 1, args.max_lr, args.min_lr, args.warmup_iters, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_ids, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"step={step:5d} lr={lr:.6e} train_loss={loss.item():.4f}")
            if getattr(args, "wandb_enabled", False):
                wandb.log({"step": step, "lr": lr, "train_loss": loss.item()})

        if step % args.eval_interval == 0 or step == 1:
            val_loss = evaluate(model, valid_ids, args.batch_size, args.context_length, device, args.eval_steps)
            print(f"step={step:5d} lr={lr:.6e} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            if getattr(args, "wandb_enabled", False):
                wandb.log({"step": step, "val_loss": val_loss})

    ckpt_path = args.work_dir / "model_last.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[info] saved model to {ckpt_path}")

    sample = generate(model, tokenizer, "Once upon a time", device=device, max_new_tokens=120, temperature=0.9)
    print("\n[sample]\n" + sample)

    if getattr(args, "wandb_enabled", False):
        wandb.log({"sample_text": sample})
        wandb.finish()


if __name__ == "__main__":
    main()
