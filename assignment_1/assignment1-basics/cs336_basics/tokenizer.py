from __future__ import annotations

from collections import Counter  # 用于计数 BPE 合并时的 pair 频次
from collections.abc import Iterable, Iterator  # 用于编码接口的流式处理
from dataclasses import dataclass  # 轻量级配置类装饰器
from pathlib import Path  # 跨平台路径处理

import regex as re  # 使用 regex 支持 Unicode 属性（比标准 re 更强大）


TokenBytes = bytes  # 类型别名：token 的字节表示
TokenId = int  # 类型别名：token 的 ID 表示
Pair = tuple[TokenBytes, TokenBytes]  # 类型别名：两个 token 的 pair，用于 BPE 合并

# GPT-2 预分词正则表达式：与 tiktoken/gpt2 完全一致的文本切分规则
# 作用：在 BPE 合并前，先将文本按照语言学逻辑切成"词"单位，降低合并复杂度
# 规则说明：
#   - 's|'t|'re|'ve|'m|'ll|'d：英文缩写
#   - | ?\p{L}+：可选空格 + 一个或多个字母
#   - | ?\p{N}+：可选空格 + 一个或多个数字
#   - | ?[^\s\p{L}\p{N}]+：可选空格 + 一个或多个非空非字母非数字字符（如标点）
#   - |\s+(?!\S)|\s+：空格序列（末尾特殊处理保留尾部空格）
GPT2_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


@dataclass(frozen=True)  # 数据类装饰器，frozen=True 表示实例创建后不可修改
class BPETokenizerConfig:
    """
    BPE tokenizer 的配置对象。

    字段说明：
    - `special_tokens`：需要被"整体保留"的特殊 token 字符串列表。
      这些 token 在编码时不应被拆分，也不应参与普通文本的 BPE 合并逻辑。

    扩展建议：
    1. 后续如需支持不同的预分词策略（例如 GPT-2 regex pretokenization），可在这里扩展配置项。
    2. 若要支持特殊 token 的策略开关（例如严格模式 / 允许未知特殊 token），也可在此新增字段。
    """

    special_tokens: list[str] | None = None  # 特殊 token 列表，默认不指定


class BPETokenizer:
    """
    字节级 BPE tokenizer。

    设计目标：
    - 与作业测试接口对齐：提供 `encode`、`decode`、`encode_iterable`。
    - 使用外部给定的 `vocab` 与 `merges` 进行推理，不在本类中直接训练。

    参数约定：
    - `vocab`: `dict[int, bytes]`，token_id -> token_bytes。
    - `merges`: `list[tuple[bytes, bytes]]`，按训练先后顺序排列。
    - `special_tokens`: `list[str] | None`，特殊 token 字符串。
    """

    def __init__(
        self,
        vocab: dict[TokenId, TokenBytes],  # 词表：id -> bytes（用于解码）
        merges: list[Pair],  # BPE 合并列表：[(left_bytes, right_bytes), ...]（按训练顺序）
        special_tokens: list[str] | None = None,  # 特殊 token 字符串列表
    ) -> None:
        """
        初始化 tokenizer。

        主要步骤：
        1. 记录 vocab 与 merges。
        2. 构建反向词表 `token_to_id`，以及 `merge_ranks`。
        3. 处理 special tokens 的编码与 id 映射。
        """
        # 防御性检查：确保词表不为空
        if not vocab:
            raise ValueError("vocab 不能为空")

        # 保存原始词表（id -> bytes 映射）
        self.vocab = dict(vocab)
        self.id_to_token = dict(vocab)  # id_to_token 直接复制词表用于解码

        # 构建反向词表：bytes -> id（用于编码时查询）
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.id_to_token.items()}

        # 保存 BPE 合并列表
        self.merges = list(merges)

        # 为了高效查询合并的优先级，构建 merge_ranks 字典
        # pair 的 rank 越小，表示该 pair 在训练中越早出现，合并优先级越高
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # 初始化特殊 token 相关的数据结构
        self.special_tokens = special_tokens or []  # 如果没传入则设为空列表

        # 将特殊 token 字符串转为 UTF-8 bytes
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]

        # 特殊 token 字符串 -> id 的映射，方便编码时快速查询
        self.special_token_to_id: dict[str, int] = {}

        # 遍历特殊 token，从词表中查找对应的 id，并建立字符串 -> id 映射
        for token_str, token_bytes in zip(self.special_tokens, self.special_token_bytes):
            # 检查特殊 token 的字节表示是否在词表中存在
            if token_bytes not in self.token_to_id:
                raise ValueError(f"special token 未在词表中找到: {token_str}")
            # 建立 special token 字符串 -> id 的快速查询表
            self.special_token_to_id[token_str] = self.token_to_id[token_bytes]

        # 为了支持最长匹配优先（避免被较短的 special token 优先匹配），
        # 将特殊 token 按长度从长到短排序
        self._special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    def encode(self, text: str) -> list[TokenId]:
        """
        将输入字符串编码为 token id 序列。

        逻辑：
        1. 先按 special token 切分。
        2. 普通片段走 `_encode_ordinary`。
        3. special token 片段直接映射为单个 id。
        """
        # 处理空字符串：直接返回空列表
        if text == "":
            return []

        # 第一步：按 special token 切分文本
        # 返回格式：[(is_special: bool, segment_text: str), ...]
        segments = self._split_by_special_tokens(text)

        # 累积编码结果
        ids: list[TokenId] = []

        # 第二步：遍历每个片段，根据类型选择编码方式
        for is_special, segment in segments:
            # 跳过空片段（不应该有，但为了防御性编程）
            if not segment:
                continue

            if is_special:
                # special token 段：直接查表获得对应的 token id
                token_id = self.special_token_to_id.get(segment)
                if token_id is None:
                    # 理论上不应该发生，因为初始化时已验证 special token
                    raise ValueError(f"未知 special token: {segment}")
                ids.append(token_id)
            else:
                # 普通文本段：调用 _encode_ordinary 进行复杂编码（含预分词 + BPE）
                ids.extend(self._encode_ordinary(segment))

        return ids

    def decode(self, ids: list[TokenId]) -> str:
        """
        将 token id 序列解码回字符串。

        逻辑：
        1. 将每个 token id 映射回对应的 bytes。
        2. 拼接所有 bytes。
        3. 按 UTF-8 解码为字符串。
        """
        # 累积每个 token id 对应的 bytes
        token_bytes_list: list[bytes] = []

        # 遍历每个 token id，将其转换为 bytes
        for token_id in ids:
            # 防御性检查：确保 token id 有效
            if token_id not in self.id_to_token:
                raise ValueError(f"非法 token id: {token_id}")

            # 查表获取 token 对应的 bytes
            token_bytes_list.append(self.id_to_token[token_id])

        # 按作业要求：对非法 UTF-8 序列使用 replacement 字符替代，而不是抛异常。
        return b"".join(token_bytes_list).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[TokenId]:
        """
        流式编码接口：输入一个字符串可迭代对象（如文件对象），按顺序 yield token id。

        适用场景：逐行/逐块读取大文件文本，不一次性加载到内存，降低内存占用。
        """
        # 遍历可迭代对象中的每个 chunk（如每行）
        for chunk in iterable:
            # 对每个 chunk 调用 encode 得到 token id 列表
            # 使用 yield from 逐个 yield token id，实现流式处理
            for token_id in self.encode(chunk):
                yield token_id

    def _encode_ordinary(self, text: str) -> list[TokenId]:
        """
        编码普通文本（不包含 special token），这是编码的核心逻辑。

        步骤：
        1. 按 GPT-2 正则切成若干预分词 token（单位：词）。
        2. 对每个预分词符号：
           a. 转成 UTF-8 bytes。
           b. 拆成单字节符号列表。
           c. 执行 BPE 合并（根据 merge_ranks 按优先级贪心合并）。
        3. 映射最终的合并后符号为 token id。
        """
        # 处理空字符串
        if text == "":
            return []

        ids: list[TokenId] = []

        # 第一步：按 GPT-2 正则切分文本，得到一个个"词"单位
        # 例如：\"Hello, world!\" -> [\"Hello\", \",\", \" world\", \"!\"]
        for piece in re.findall(GPT2_SPLIT_PATTERN, text):
            # 将预分词符号转为 UTF-8 bytes
            piece_bytes = piece.encode("utf-8")

            # 初始化符号列表：每个字节作为一个初始符号
            # 例如：\"Hello\" 的 UTF-8 字节是 [72, 101, 108, 108, 111]，
            # 初始符号列表为 [b'H', b'e', b'l', b'l', b'o']
            symbols = [bytes([b]) for b in piece_bytes]

            # 第二步：应用 BPE 合并规则
            # 根据已学习的 merges，贪心地合并相邻符号
            merged_symbols = self._apply_bpe_merges(symbols)

            # 第三步：将最终符号映射为 token id
            for sym in merged_symbols:
                # 从词表中查询符号对应的 id
                token_id = self.token_to_id.get(sym)
                if token_id is None:
                    # 理论上不应该有合并后的符号不在词表中的情况
                    raise ValueError(f"词表中缺少 token: {sym}")
                ids.append(token_id)

        return ids

    def _split_by_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        """
        将输入文本切成 [(是否为special_token, 片段文本)] 的格式。

        使用\"最长匹配优先\"的策略处理有重叠关系的 special token
        （例如 \"<|a|>\" 和 \"<|a|><|a|>\" 不会被部分匹配）。
        """
        # 若没有特殊 token，直接返回整个文本作为普通片段
        if not self._special_tokens_sorted:
            return [(False, text)]

        # 累积切分结果
        segments: list[tuple[bool, str]] = []
        i = 0  # 当前扫描位置
        text_len = len(text)

        # 线性扫描整个文本
        while i < text_len:
            match_token = None

            # 尝试匹配最长的 special token（自动最长匹配）
            # 由于 _special_tokens_sorted 已按长度从长到短排序，第一个匹配就是最长的
            for token in self._special_tokens_sorted:
                if text.startswith(token, i):  # 检查当前位置是否以 token 开头
                    match_token = token
                    break

            if match_token is None:
                # 当前位置不是 special token 的起始，向前推进
                # 直到遇到 special token 的起始位置或文本结束
                j = i + 1
                while j < text_len and not any(text.startswith(t, j) for t in self._special_tokens_sorted):
                    j += 1
                # 添加普通文本片段 [(False, \"...\")]
                segments.append((False, text[i:j]))
                i = j
            else:
                # 找到一个 special token，添加 [(True, special_token_str)]
                segments.append((True, match_token))
                # 跳过这个 special token
                i += len(match_token)

        return segments

    def _apply_bpe_merges(self, symbols: list[TokenBytes]) -> list[TokenBytes]:
        """
        对初始符号序列应用 BPE merges，返回合并后的符号序列。

        实现策略：\"逐轮找最优 pair\"的贪心算法
        - 每轮找到当前可合并且优先级最高（rank 最小）的 pair。
        - 执行一次全局合并。
        - 重复直到无可合并 pair。

        这是朴素实现，逻辑清晰，易于验证和调试。
        """
        # 防御性检查：符号少于 2 个无法合并
        if len(symbols) < 2:
            return symbols

        # 迭代合并，直到无可合并的 pair
        while True:
            best_pair = None  # 本轮最优合并 pair
            best_rank = None  # 本轮最优 pair 的 rank

            # 第一步：扫描所有相邻 pair，找到 rank 最小的（优先级最高）
            for i in range(len(symbols) - 1):
                # 取相邻两个符号组成 pair
                pair = (symbols[i], symbols[i + 1])
                # 在 merge_ranks 中查询该 pair 的 rank
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    # 该 pair 不在训练的 merges 中，跳过
                    continue
                # 更新最优 pair（rank 越小优先级越高）
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 如果没找到任何可合并的 pair，停止迭代
            if best_pair is None:
                break

            # 第二步：执行本轮合并
            # 遍历符号序列，将所有匹配的 pair 合并为一个符号
            merged: list[TokenBytes] = []
            i = 0
            while i < len(symbols):
                # 检查当前位置是否是要合并的 pair 的起始
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    # 合并：连接两个符号的字节
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2  # 跳过两个符号
                else:
                    # 不是目标 pair，保持原样
                    merged.append(symbols[i])
                    i += 1

            # 更新符号列表，进行下一轮迭代
            symbols = merged

        return symbols


def train_bpe(
    input_path: str | Path,  # 输入语料文件路径
    vocab_size: int,  # 目标词表大小（包含 special tokens）
    special_tokens: list[str],  # 特殊 token 列表（如 \"<|endoftext|>\"）
) -> tuple[dict[TokenId, TokenBytes], list[Pair]]:
    """
    从语料文件训练字节级 BPE tokenizer，返回 `(vocab, merges)`。

    核心设计原理：
    - 初始符号分布：256 个字节 + special tokens。
    - 迭代策略：逐轮找最高频 pair，生成新符号，合并所有出现，更新频次表。
    - 终止条件：vocab 达到目标大小，或没有可合并 pair。

    使用 GPT-2 的正则进行预分词，控制初始符号复杂度。
    特殊 token 作为原子单位加入词表，不参与 BPE 合并。
    """
    # 转换为 Path 对象，便于跨平台路径操作
    input_path = Path(input_path)
    # 检查文件存在性
    if not input_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {input_path}")

    # 计算最少需要的词表大小
    min_vocab = 256 + len(special_tokens)  # 256 个字节 + special tokens
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size 太小，至少需要 {min_vocab}")

    # 读取整个语料文件
    text = input_path.read_text(encoding="utf-8")

    # 初始化词表：ID 0-255 对应单字节
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 将 special tokens 追加到词表中
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        # 避免重复添加（若 special token 的 bytes 恰好等于某个单字节，跳过）
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # 构建词频表
    # key：tuple[bytes, ...] - 符号序列（例如 (b'H', b'e', b'l', b'l', b'o')）
    # value：int - 该序列在语料中的出现次数
    words: Counter[tuple[bytes, ...]] = Counter()

    # 为了处理 special tokens，先按长度从长到短排序（最长匹配优先）
    if special_tokens:
        special_sorted = sorted(special_tokens, key=len, reverse=True)
    else:
        special_sorted = []

    def split_special(text_chunk: str) -> list[tuple[bool, str]]:
        # 辅助函数：将文本按 special token 切分
        # 返回 [(is_special: bool, segment: str), ...]
        if not special_sorted:
            return [(False, text_chunk)]  # 无 special token，整个是普通文本

        segments: list[tuple[bool, str]] = []
        i = 0
        while i < len(text_chunk):
            match_token = None
            # 尝试匹配最长的 special token
            for token in special_sorted:
                if text_chunk.startswith(token, i):
                    match_token = token
                    break
                    
            if match_token is None:
                # 普通字符，向前扫描直到遇到 special token
                j = i + 1
                while j < len(text_chunk) and not any(text_chunk.startswith(t, j) for t in special_sorted):
                    j += 1
                segments.append((False, text_chunk[i:j]))
                i = j
            else:
                # special token
                segments.append((True, match_token))
                i += len(match_token)

        return segments

    # 遍历整个语料，构建词频表
    for is_special, segment in split_special(text):
        # 跳过空片段
        if not segment:
            continue

        if is_special:
            # special token 段：作为原子单位，其 bytes 形式作为一个符号
            # 统计如 (b'<|endoftext|>',) 这样的单符号序列
            words[(segment.encode("utf-8"),)] += 1
            continue

        # 普通文本段：按 GPT-2 正则切成词，再统计相邻 pair
        for piece in re.findall(GPT2_SPLIT_PATTERN, segment):
            # 将词转为 UTF-8 bytes
            piece_bytes = piece.encode("utf-8")
            # 初始化为单字节符号序列
            # 例如 \"hello\" -> (b'h', b'e', b'l', b'l', b'o')
            word = tuple(bytes([b]) for b in piece_bytes)
            if word:  # 防止空序列
                # 统计该词序列的出现次数
                words[word] += 1

    # BPE 合并列表，按合并顺序记录
    merges: list[Pair] = []

    def merge_word(word: tuple[bytes, ...], pair: Pair) -> tuple[bytes, ...]:
        # 辅助函数：对一个词应用单个 merge 操作
        # 将词序列中的所有 pair 合并为一个符号
        merged: list[bytes] = []
        i = 0
        while i < len(word):
            # 检查当前位置是否匹配要合并的 pair
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                # 合并：两个符号连接为一个
                merged.append(word[i] + word[i + 1])
                i += 2  # 跳过两个符号
            else:
                # 不匹配，保持原样
                merged.append(word[i])
                i += 1
        return tuple(merged)

    # BPE 主循环：迭代合并直到达到目标词表大小
    while len(vocab) < vocab_size:
        # 计算所有相邻 pair 的频次
        pair_counts: Counter[Pair] = Counter()
        for word, freq in words.items():
            # 词序列长度 < 2 无法提取 pair
            if len(word) < 2:
                continue
            # 统计该词中的所有相邻 pair
            for i in range(len(word) - 1):
                # pair 的频次累加该词的频次
                pair_counts[(word[i], word[i + 1])] += freq

        # 如果没有 pair 可合并，提前终止
        if not pair_counts:
            break

        # 选择最高频 pair
        # 为了确定性（在平局时行为不变），用 pair 本身作为次级键排序
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        # 记录本次合并
        merges.append(best_pair)

        # 生成新符号（两个旧符号连接）
        new_token = best_pair[0] + best_pair[1]
        # 如果新符号在词表中不存在，追加到词表
        if new_token not in vocab.values():
            vocab[len(vocab)] = new_token

        # 更新所有词：将 best_pair 合并为新符号
        new_words: Counter[tuple[bytes, ...]] = Counter()
        for word, freq in words.items():
            # 对每个词应用当前的 merge 操作
            merged_word = merge_word(word, best_pair)
            new_words[merged_word] += freq
        words = new_words

    # 返回最终的词表和合并列表
    return vocab, merges
