from __future__ import annotations

from collections import Counter
from pathlib import Path


# ============================================================================
# 类型定义
# ============================================================================

TokenBytes = bytes  # token 的字节表示
TokenId = int  # token 的 ID 表示
Pair = tuple[TokenBytes, TokenBytes]  # 两个 token 的 pair，用于 BPE 合并


# ============================================================================
# 辅助函数
# ============================================================================

def _apply_bpe_merges(
    symbols: list[TokenBytes],
    merge_ranks: dict[Pair, int]
) -> list[TokenBytes]:
    """
    对初始符号序列应用 BPE merges，返回合并后的符号序列。

    参数：
    - symbols: 初始符号列表（通常是单字节 token 列表）
    - merge_ranks: pair -> rank 的映射，rank 越小优先级越高

    TODO（实现建议）：
    1. 每轮找出当前序列中 rank 最小（最先训练出的）可合并 pair。
    2. 执行一次合并，将该 pair 合并为一个新符号。
    3. 重复直到不存在可合并 pair 为止。
    4. 返回合并后的符号列表。

    示例：
    如果 symbols = [b'h', b'e', b'l', b'l', b'o']
    并且 merge_ranks = {(b'l', b'l'): 0}
    应该返回 [b'h', b'e', b'll', b'o']
    """
    raise NotImplementedError


# ============================================================================
# 核心函数：encode 和 decode
# ============================================================================

def encode(
    text: str,
    vocab: dict[TokenId, TokenBytes],
    merges: list[Pair],
    special_tokens: list[str] | None = None
) -> list[TokenId]:
    """
    将输入字符串编码为 token id 序列。

    参数：
    - text: 要编码的文本
    - vocab: 词表，token_id -> token_bytes
    - merges: BPE 合并列表，按训练先后顺序排列
    - special_tokens: 特殊 token 列表（可选）

    返回：
    - token id 列表

    TODO（实现步骤）：
    1. 从 vocab 构建反向映射 token_to_id（bytes -> id）
    2. 从 merges 构建 merge_ranks（pair -> rank，rank 越小优先级越高）
    3. 如果有 special_tokens，先将文本按 special token 切分
       - 对于普通文本段：
         a. 转为 UTF-8 bytes
         b. 拆成单字节符号列表：[bytes([b]) for b in text_bytes]
         c. 调用 _apply_bpe_merges 应用 BPE 合并
         d. 将符号映射为 token id
       - 对于 special token 段：直接映射为对应的 token id
    4. 返回完整的 token id 列表

    提示：
    - 空字符串应返回空列表
    - special_tokens 的切分建议使用"最长匹配优先"
    - 确保所有 token bytes 都在词表中
    """
    raise NotImplementedError


def decode(
    ids: list[TokenId],
    vocab: dict[TokenId, TokenBytes]
) -> str:
    """
    将 token id 序列解码回字符串。

    参数：
    - ids: token id 列表
    - vocab: 词表，token_id -> token_bytes

    返回：
    - 解码后的字符串

    TODO（实现步骤）：
    1. 遍历每个 token id
    2. 从 vocab 中查找对应的 token bytes
    3. 将所有 bytes 拼接成一个完整的 bytes 对象
    4. 使用 UTF-8 解码为字符串并返回

    提示：
    - 需要处理非法 token id（不在词表中的情况），可以抛出 ValueError
    - 使用 b"".join(token_bytes_list).decode("utf-8") 进行解码
    """
    raise NotImplementedError


# ============================================================================
# 核心函数：train_bpe（训练 BPE tokenizer）
# ============================================================================

def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[TokenId, TokenBytes], list[Pair]]:
    """
    从语料文件训练字节级 BPE tokenizer，返回 (vocab, merges)。

    参数：
    - input_path: 输入语料文件路径
    - vocab_size: 目标词表大小（包含 special tokens）
    - special_tokens: 特殊 token 列表（如 ["<|endoftext|>"]）

    返回：
    - vocab: dict[token_id, token_bytes]，词表
    - merges: list[(left_bytes, right_bytes)]，BPE 合并列表，按训练顺序

    TODO（实现步骤）：

    第一步：初始化词表
    1. 读取语料文件（使用 UTF-8 编码）
       text = Path(input_path).read_text(encoding="utf-8")
    2. 初始化词表：包含 256 个单字节 token (ID 0-255)
       vocab = {i: bytes([i]) for i in range(256)}
    3. 将 special_tokens 追加到词表中
       - 将每个 special token 转为 UTF-8 bytes
       - 添加到 vocab 中，ID 从 256 开始
    
    第二步：构建初始词频表
    1. 将语料文本转为 UTF-8 bytes
    2. 将整个 bytes 序列拆成单字节序列（或按词切分后再拆）
       例如：text_bytes = text.encode("utf-8")
             word = tuple(bytes([b]) for b in text_bytes)
    3. 使用 Counter 统计词序列的频次
       words: Counter[tuple[bytes, ...]] = Counter()
       words[word] = 1  # 如果是整个文本作为一个"词"

    第三步：迭代合并
    while len(vocab) < vocab_size:
        1. 统计所有相邻 pair 的频次
           pair_counts: Counter[Pair] = Counter()
           for word, freq in words.items():
               if len(word) < 2:
                   continue
               for i in range(len(word) - 1):
                   pair_counts[(word[i], word[i + 1])] += freq
        
        2. 选择最高频 pair
           if not pair_counts:
               break  # 没有可合并的 pair
           best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        
        3. 记录合并
           merges.append(best_pair)
           new_token = best_pair[0] + best_pair[1]
           if new_token not in vocab.values():
               vocab[len(vocab)] = new_token
        
        4. 更新词频表
           new_words: Counter[tuple[bytes, ...]] = Counter()
           for word, freq in words.items():
               merged_word = _merge_word(word, best_pair)
               new_words[merged_word] += freq
           words = new_words

    第四步：返回结果
    return vocab, merges

    提示：
    - 最小词表大小 = 256 + len(special_tokens)
    - 使用 Counter 统计频次会更高效
    - 需要实现辅助函数 _merge_word(word, pair) 来合并词中的 pair
      def _merge_word(word, pair):
          result = []
          i = 0
          while i < len(word):
              if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                  result.append(word[i] + word[i + 1])
                  i += 2
              else:
                  result.append(word[i])
                  i += 1
          return tuple(result)
    - 为了效率，不要每次都扫描整个字符串，而是维护词序列的表示
    """
    input_path = Path(input_path)
    
    # 检查文件存在性
    if not input_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {input_path}")
    
    # 检查词表大小是否足够
    min_vocab = 256 + len(special_tokens)
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size 太小，至少需要 {min_vocab}")
    
    # TODO: 在这里实现训练逻辑
    # 1. 读取语料
    # 2. 初始化词表
    # 3. 构建词频表
    # 4. 迭代合并
    # 5. 返回 vocab 和 merges
    
    raise NotImplementedError


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：训练 BPE
    # vocab, merges = train_bpe(
    #     input_path="corpus.txt",
    #     vocab_size=300,
    #     special_tokens=["<|endoftext|>"]
    # )
    
    # 示例：使用训练好的 vocab 和 merges 进行编码
    # text = "Hello world!"
    # ids = encode(text, vocab, merges, special_tokens=["<|endoftext|>"])
    # print(f"编码结果: {ids}")
    
    # 示例：解码
    # decoded_text = decode(ids, vocab)
    # print(f"解码结果: {decoded_text}")
    
    pass
