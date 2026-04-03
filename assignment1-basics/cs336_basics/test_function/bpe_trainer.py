import os
import regex as re
import multiprocessing
from typing import BinaryIO
from collections import Counter, defaultdict

# ==========================================
# 1. 全局配置与正则定义
# ==========================================

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_REGEX = re.compile(PATTERN)

# ==========================================
# 2. 多进程预分词辅助函数
# ==========================================

def process_chunk_with_special_tokens(chunk_text: str, special_tokens: list[str], pretoken_regex: re.Pattern) -> Counter:
    split_pattern = "|".join(map(re.escape, special_tokens)) if special_tokens else "(?!)"
    text_segments = re.split(split_pattern, chunk_text)
    chunk_counts = Counter()
    
    for segment in text_segments:
        if not segment: 
            continue
        for match in pretoken_regex.finditer(segment):
            pre_token = match.group()
            byte_token = pre_token.encode("utf-8")
            chunk_counts[byte_token] += 1
            
    return chunk_counts

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    if not split_special_token:
        # 如果没有特殊的 token 用于切分边界，则退化为按字节硬切分
        # 注意：在真实的严格环境中，这有极小概率切断多字节字符，但为了兼容性保留
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size
        return sorted(set(boundaries))

    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def worker_task(file_path: str, start: int, end: int, special_tokens: list[str]) -> Counter:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    return process_chunk_with_special_tokens(chunk_text, special_tokens, PRETOKEN_REGEX)

def run_parallel_pretokenization(input_path: str, special_tokens: list[str]) -> Counter:
    """内部封装的并行执行逻辑"""
    num_workers = multiprocessing.cpu_count()
    num_tasks = num_workers * 4

    # 选择第一个 special token 作为物理切分边界（如果没有，传空字节）
    boundary_token = special_tokens[0].encode('utf-8') if special_tokens else b""
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_tasks, boundary_token)

    with multiprocessing.Pool(processes=num_workers) as pool:
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        partial_counts = pool.starmap(worker_task, tasks)

    global_counts = Counter()
    for local_count in partial_counts:
        global_counts.update(local_count)
        
    return global_counts


# ==========================================
# 3. 核心大作业交付函数: train_bpe
# ==========================================

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 Byte-level BPE Tokenizer.
    包含工业级优化：并行预分词与基于缓存的增量更新。
    """
    # 1. 运行并行预分词，获取语料库全局词频
    print("开始预分词与频率统计...")
    global_counts = run_parallel_pretokenization(input_path, special_tokens)

    # 2. 初始化词表与合并参数
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_idx = 256
    merges = []
    
    # 计算需要进行多少次合并操作
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small to fit base bytes and special tokens.")

    # 3. 初始化增量更新数据结构
    # splits: 记录每个 pre-token 当前被拆解成了什么 token 序列
    splits = {word: [bytes([b]) for b in word] for word in global_counts}
    # pair_counts: 全局 token 对出现的频率
    pair_counts = Counter()
    # pair_to_words: 倒排索引，记录哪些单词包含了特定的 token 对
    pair_to_words = defaultdict(set)

    print("构建初始字节对缓存...")
    for word, split in splits.items():
        freq = global_counts[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # 4. 执行 BPE 迭代合并
    print(f"开始执行合并 (目标 {num_merges} 次)...")
    for i in range(num_merges):
        if not pair_counts:
            break  # 已经没有可以合并的 pair 了

        # 找到频率最高的 pair
        # 遇到平局时，Python 的元组比较 (count, pair) 会自动以字典序较大的 pair 胜出
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        # 记录本次合并
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        # 加入词表
        vocab[vocab_idx] = new_token
        vocab_idx += 1

        # ==== 核心优化：增量更新受影响的单词 ====
        words_to_update = list(pair_to_words[best_pair])
        
        # 将被合并的 best_pair 从全局统计中移除
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_update:
            split = splits[word]
            freq = global_counts[word]

            # 步骤 A：将该单词中所有旧的 pair 计数扣除
            for j in range(len(split) - 1):
                p = (split[j], split[j+1])
                if p != best_pair: # best_pair 已经被彻底移除了，不需要再扣减
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(word)

            # 步骤 B：执行真正的序列合并替换
            new_split = []
            idx = 0
            while idx < len(split):
                if idx < len(split) - 1 and split[idx] == best_pair[0] and split[idx+1] == best_pair[1]:
                    new_split.append(new_token)
                    idx += 2
                else:
                    new_split.append(split[idx])
                    idx += 1
            splits[word] = new_split

            # 步骤 C：为产生的新序列重新加上 pair 计数
            for j in range(len(new_split) - 1):
                p = (new_split[j], new_split[j+1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)
                
        if (i + 1) % 1000 == 0:
            print(f"  已完成合并: {i + 1} / {num_merges}")

    # 5. 将 Special Tokens 添加到最终词表末尾
    for st in special_tokens:
        vocab[vocab_idx] = st.encode("utf-8")
        vocab_idx += 1

    print("BPE 训练完成！")
    return vocab, merges


# ==========================================
# 测试/执行入口
# ==========================================
if __name__ == "__main__":
    file_path = "/home/inferior/Projects/CS336/data/TinyStoriesV2-GPT4-train.txt"
    # 根据你的作业要求，适当设置词表大小。比如 GPT-2 的词表大约是 50257
    target_vocab_size = 10000 
    special_tks = ["<|endoftext|>"]
    
    vocab, merges = train_bpe(
        input_path=file_path, 
        vocab_size=target_vocab_size, 
        special_tokens=special_tks
    )
    
    print(f"\n最终合并列表长度: {len(merges)}")
    print(f"最终词表大小: {len(vocab)}")
    
    # 打印前 5 个发生的合并看看效果
    print("\n最初的 5 次 Merge:")
    for i in range(min(500, len(merges))):
        p1, p2 = merges[i]
        print(f"  {p1} + {p2} -> {p1 + p2}")