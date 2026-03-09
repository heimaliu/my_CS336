from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.model import Embedding, Linear, MultiHeadAttentionWithRoPE, RMSNorm, SwiGLU
from cs336_basics.nn_utils import cross_entropy, get_batch, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.schedule import get_lr_cosine_schedule
from cs336_basics.tokenizer import BPETokenizer, train_bpe


class TrainableTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, rope_theta: float, eps: float = 1e-5):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps)
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, rope_theta, context_length)
        self.ln2 = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions=positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TrainableTransformerLM(nn.Module):
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
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TrainableTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
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
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--valid_path", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("runs/tinystories_simple"))

    parser.add_argument("--max_train_chars", type=int, default=2_000_000)
    parser.add_argument("--max_valid_chars", type=int, default=400_000)
    parser.add_argument("--vocab_size", type=int, default=5000)

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tok_cache = args.work_dir / "tokenizer.pkl"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[info] device={device}")

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
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

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

        if step % args.eval_interval == 0 or step == 1:
            val_loss = evaluate(model, valid_ids, args.batch_size, args.context_length, device, args.eval_steps)
            print(f"step={step:5d} lr={lr:.6e} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    ckpt_path = args.work_dir / "model_last.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[info] saved model to {ckpt_path}")

    sample = generate(model, tokenizer, "Once upon a time", device=device, max_new_tokens=120, temperature=0.9)
    print("\n[sample]\n" + sample)


if __name__ == "__main__":
    main()
