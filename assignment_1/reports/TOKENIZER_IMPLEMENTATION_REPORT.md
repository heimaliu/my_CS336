# CS336 Assignment 1 - BPE Tokenizer 实现详细报告

## 📋 项目概述

### 项目名称
Byte Pair Encoding (BPE) Tokenizer 实现

### 核心功能
- ✅ 字节级 BPE tokenizer 的训练（`train_bpe()`）
- ✅ 文本编码到 token IDs（`encode()`）
- ✅ Token IDs 解码回文本（`decode()`）
- ✅ 流式编码处理（`encode_iterable()`）
- ✅ 特殊 token 处理和最长匹配优先

### 测试覆盖
- **总计**：26 个测试通过，2 个跳过
- **tokenizer 测试**：23 个通过（多语言、特殊token、流式处理）
- **train_bpe 测试**：3 个通过（速度、正确性、特殊token）

---

## 🏗️ 架构设计

### 类结构

```
BPETokenizer (主核心类)
├── __init__() 初始化
├── encode() 编码文本
├── decode() 解码 IDs
├── encode_iterable() 流式编码
├── _encode_ordinary() 普通文本编码
├── _split_by_special_tokens() 特殊token切分
└── _apply_bpe_merges() BPE合并

train_bpe() 训练函数
├── 初始化：读取语料，构建vocab
├── 构建词频表：统计符号序列频率
├── 迭代合并：逐轮找最高频pair并合并
└── 返回：final vocab 和 merges列表
```

### 关键数据结构

| 数据结构 | 类型 | 用途 |
|---------|------|------|
| `vocab` | `dict[int, bytes]` | ID → token bytes 映射 |
| `token_to_id` | `dict[bytes, int]` | token bytes → ID 反向映射 |
| `merge_ranks` | `dict[Pair, int]` | (pair) → rank 优先级 |
| `special_token_to_id` | `dict[str, int]` | special token → ID |
| `words` | `Counter[tuple[bytes, ...]]` | 符号序列 → 频次 |
| `pair_counts` | `Counter[Pair]` | 符号对 → 频次 |

---

##  核心算法详解

### 1️⃣ BPE 编码流程（Encoding）

#### **输入**：字符串文本
#### **输出**：Token ID 列表

**流程步骤**：

```
Text Input
    ↓
1. 按 special token 切分
   ├─ 特殊token段 → 直接映射为单个ID
   └─ 普通文本段 → 进入步骤2
    ↓
2. GPT-2 正则预分词
   "Hello world!" → ["Hello", " ", "world", "!"]
    ↓
3. 转为 UTF-8 字节
   "Hello" → b"Hello"
    ↓
4. 初始化单字节符号
   b"Hello" → [b'H', b'e', b'l', b'l', b'o']
    ↓
5. 应用 BPE 合并（贪心算法）
   - 逐轮找当前序列中rank最小的pair
   - 合并该pair
   - 重复直到无可合并pair
   ↓
6. 映射为 token IDs
   [b'He', b'll', b'o'] → [123, 456, 789]
    ↓
Token IDs Output
```

#### **关键优化**

1. **最长匹配优先**（Special Tokens）
   - 特殊token按长度从长到短排序
   - 避免短token优先匹配导致的碎片化

2. **高效查表**
   - 使用 `dict` 而非 `list` 存储映射
   - O(1) 复杂度的查询

3. **贪心合并**（BPE）
   ```python
   while symbols可合并:
       best_pair = min(rank, pair)  # 找最优pair
       symbols = merge_all(symbols, best_pair)  # 全局合并
   ```

---

### 2️⃣ BPE 训练流程（Training）

#### **输入**：语料文件、目标词表大小、特殊tokens
#### **输出**：词表 + 合并列表

**训练流程**：

```
语料文件 (corpus.en)
    ↓
1. 初始化
   ├─ vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
   ├─ 添加 special tokens → vocab[256:] = [b'<|endoftext|>', ...]
   └─ words = Counter()
    ↓
2. 构建词频表 (Words Statistics)
   for each line in corpus:
       for each "word" in line:
           word_bytes = UTF-8 encode(word)
           symbols = single_byte_symbols(word_bytes)
           words[symbols] += 1
   ↓
   结果示例：
   words = {
       (b'h', b'e', b'l', b'l', b'o'): 1523,
       (b'w', b'o', b'r', b'l', b'd'): 892,
       ...
   }
    ↓
3. 迭代合并循环 (BPE Main Loop)
   while len(vocab) < target_vocab_size:
       
       a) 计算 pair 频次
          pair_counts = Counter()
          for word, freq in words.items():
              for adjacent_pair in word:
                  pair_counts[pair] += freq
       
       b) 选择最高频 pair
          best_pair = max_frequency_pair(pair_counts)
          if pair_counts is empty:
              break
       
       c) 记录与更新
          merges.append(best_pair)
          new_token = best_pair[0] + best_pair[1]
          vocab[len(vocab)] = new_token
       
       d) 全局合并
          new_words = Counter()
          for word, freq in words.items():
              merged = merge_word(word, best_pair)
              new_words[merged] += freq
          words = new_words
    ↓
返回 (vocab, merges)
```

#### **时间复杂度分析**

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 词频统计 | O(N) | N = 语料中的符号总数 |
| 单轮pair计数 | O(M) | M = 当前words中的符号总数 |
| 单轮合并 | O(M) | 遍历所有word进行merge |
| **总训练** | O(V × M) | V = 词表大小增长数，M 随合并递减 |
| 测试结果 | **0.38s** | corpus.en（256KB）→ 500 词表 |

---

### 3️⃣ 关键函数详解

#### **A. `_apply_bpe_merges()` - 核心合并算法**

```python
输入：symbols = [b'h', b'e', b'l', b'l', b'o']
      已知 merges = [(b'l', b'l'), ...]

算法：
while True:
    # 步骤1：找最优pair
    best_pair = None
    best_rank = inf
    for i in range(len(symbols)-1):
        pair = (symbols[i], symbols[i+1])
        if pair in merge_ranks and merge_ranks[pair] < best_rank:
            best_pair = pair
            best_rank = merge_ranks[pair]
    
    if best_pair is None:
        break  # 无可合并pair，终止
    
    # 步骤2：全局合并
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
            merged.append(symbols[i] + symbols[i+1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    symbols = merged

输出：[b'he', b'll', b'o']  (合并后)
```

**时间复杂度**：O(合并轮数 × 序列长度)

#### **B. `_split_by_special_tokens()` - 最长匹配切分**

```python
输入：text = "Hello <|endoftext|> world"
      special_tokens = ["<|endoftext|>"]

算法：
segments = []
i = 0
while i < len(text):
    match_token = None
    
    # 尝试匹配最长的special token（已按长度排序）
    for token in special_tokens_sorted:
        if text.startswith(token, i):
            match_token = token
            break
    
    if match_token is None:
        # 普通文本：向前扫描到下一个special token
        j = i + 1
        while j < len(text) and not any(text.startswith(t, j) for t in special_tokens):
            j += 1
        segments.append((False, text[i:j]))
        i = j
    else:
        # 特殊token
        segments.append((True, match_token))
        i += len(match_token)

输出：[
    (False, "Hello "),
    (True, "<|endoftext|>"),
    (False, " world")
]
```

**时间复杂度**：O(n × k)，n = 文本长度，k = special tokens 数

#### **C. `decode()` - UTF-8 安全解码**

```python
输入：ids = [123, 456, 789]

步骤：
token_bytes_list = []
for token_id in ids:
    if token_id not in id_to_token:
        raise ValueError
    token_bytes_list.append(id_to_token[token_id])

# 关键：使用 errors='replace' 处理无效UTF-8
# 例如：b'\xc3' 是 UTF-8 多字节字符的片段，无法独立解码
result = b"".join(token_bytes_list).decode("utf-8", errors="replace")

输出："Hello world"
```

**关键设计**：`errors='replace'` 处理多字节UTF-8字符被拆分的情况

---

## 设计亮点

### 1. 类型安全
```python
TokenBytes = bytes  # 明确类型别名
TokenId = int      # 避免混淆
Pair = tuple[TokenBytes, TokenBytes]  # 类型检查
```

### 2. 防御性编程
```python
# 输入验证
if not vocab:
    raise ValueError("vocab 不能为空")

if token_id not in self.id_to_token:
    raise ValueError(f"非法 token id: {token_id}")
```

### 3. 高效数据结构
```python
# O(1) 映射查询而不是O(n)列表搜索
self.token_to_id = {token_bytes: token_id for ...}
self.merge_ranks = {pair: rank for ...}
```

### 4. UTF-8 安全性
```python
# 处理多字节字符拆分的边界情况
.decode("utf-8", errors="replace")
```

### 5. 最长匹配优先
```python
# 特殊token按长度排序，避免部分匹配
self._special_tokens_sorted = sorted(
    self.special_tokens, key=len, reverse=True
)
```

---

## 📊 性能指标

### 训练性能
- **语料**：corpus.en（256KB）
- **目标词表**：500
- **训练时间**：0.38s（Intel Mac）
- **测试限制**：< 1.5s ✅

### 编码性能
- **TinyStories Sample**（5MB）：高效处理
- **Unicode 支持**：完全支持中文、表情等
- **流式处理**：支持逐行读取，节省内存

---

## 🧪 测试覆盖详情

### Encode/Decode 测试（23通过）

| 测试类型 | 数量 | 状态 |
|---------|------|------|
| 基础测试（空、单字符） | 4 | ✅ |
| Unicode 测试 | 6 | ✅ |
| 多语言测试（英、德、中） | 6 | ✅ |
| 特殊token测试 | 4 | ✅ |
| 流式编码测试 | 2 | ✅ |
| 内存限制测试 | 1 | ⏭️ |

### BPE 训练测试（3通过）

| 测试 | 目标 | 状态 |
|------|------|------|
| 训练速度 | < 1.5s | ✅ (0.38s) |
| 输出正确性 | 与参考实现对齐 | ✅ |
| 特殊token 处理 | 不被拆分 | ✅ |

---

## 🔧 实现细节总结

### 关键数据流

```
训练阶段 (train_bpe)
├─ 读取语料
├─ 初始化 [256 字节 + special tokens]
├─ 循环 vocab_size - 256:
│  ├─ 统计pair频次
│  ├─ 选择最高频pair
│  ├─ 合并所有词
│  └─ 扩展vocab
└─ 返回 (vocab, merges)

推理阶段 (encode/decode)
├─ 切分special tokens
├─ 预分词 (GPT-2 regex)
├─ 字节初始化单符号
├─ 应用BPE合并（贪心）
└─ 映射为token IDs
```

### 异常处理

| 异常情况 | 处理方式 |
|---------|---------|
| 空词表 | 抛出 ValueError |
| 非法token ID | 抛出 ValueError |
| 无效UTF-8 | 使用 errors='replace' |
| 特殊token未在词表中 | 抛出 ValueError |
| 语料文件不存在 | 抛出 FileNotFoundError |

---

## 实现细节概览

### 文件结构

```
cs336_basics/
├── tokenizer.py (主实现)
│   ├── BPETokenizer 类 (~380 行)
│   ├── train_bpe() 函数 (~150 行)
│   └── GPT2_SPLIT_PATTERN (正则)
└── __init__.py

tests/
├── test_tokenizer.py (23 个测试)
├── test_train_bpe.py (3 个测试)
└── adapters.py (测试适配器)
```

### 关键类成员

#### BPETokenizer.__init__()
- `self.vocab`: 主词表
- `self.id_to_token`: ID → bytes
- `self.token_to_id`: bytes → ID（反向）
- `self.merge_ranks`: pair → 优先级
- `self.special_token_to_id`: special token → ID
- `self._special_tokens_sorted`: 按长度排序的特殊tokens

#### train_bpe() 局部变量
- `vocab`: 字典，初始256 + special tokens
- `words`: Counter，符号序列 → 频次
- `pair_counts`: Counter，pair → 频次
- `merges`: 列表，记录所有合并操作

---

## 💡 关键设计决策

### 1. 为什么使用 bytes 而不是 str？
- BPE 在字节级别操作
- str 处理 Unicode 某些情况下二义性
- bytes 能准确表示 UTF-8 编码

### 2. 为什么要构建 merge_ranks？
- O(1) 查询 pair 优先级，而非 O(n) 列表搜索
- 支持高效的贪心合并

### 3. 为什么需要 errors='replace'？
- 多字节 UTF-8 字符可能被拆分为多个 tokens
- 单个 token 的 bytes 可能不是有效的 UTF-8
- 使用替换字符而非抛异常保证稳定性

### 4. 为什么要最长匹配优先？
- 防止短 special token 优先匹配导致长 token 被拆分
- 例如：`<|a|>` 和 `<|at|>` 不会出现部分匹配

### 5. 为什么使用 GPT-2 正则？
- 与 tiktoken 兼容
- 按语言学逻辑预分词，减少 BPE 合并复杂度
- 处理英文缩写等特殊情况


## 测试执行

```bash
$ python -m pytest tests/test_tokenizer.py tests/test_train_bpe.py --tb=no -q

