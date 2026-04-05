import os
import regex as re
import multiprocessing
import heapq  # 🚀 新增：用于维护大顶堆
from typing import BinaryIO
from collections import Counter, defaultdict

# ==========================================
# 1. 全局配置与正则定义
# ==========================================

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_REGEX = re.compile(PATTERN)

# 🚀 新增：自定义堆元素类（使用 __slots__ 极致节省内存）
class HeapItem:
    __slots__ = ['count', 'pair']
    
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair
        
    def __lt__(self, other):
        # heapq 是小顶堆，每次 pop 最小的。
        # 我们希望最先 pop 出的是 count 最大的；如果 count 一样，取字典序最大的 pair
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

# ==========================================
# 2. 多进程预分词辅助函数 (保持不变，已足够快)
# ==========================================

def process_chunk_with_special_tokens(chunk_text: str, special_tokens: list[str], pretoken_regex: re.Pattern) -> Counter:
    split_pattern = "|".join(map(re.escape, special_tokens)) if special_tokens else "(?!)"
    text_segments = re.split(split_pattern, chunk_text)
    chunk_counts = Counter()
    for segment in text_segments:
        if not segment: continue
        for match in pretoken_regex.finditer(segment):
            pre_token = match.group()
            byte_token = pre_token.encode("utf-8")
            chunk_counts[byte_token] += 1
    return chunk_counts

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    if not split_special_token:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size
        return sorted(set(boundaries))

    assert isinstance(split_special_token, bytes)
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
    num_workers = multiprocessing.cpu_count()
    num_tasks = num_workers * 4
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
    print("开始预分词与频率统计...")
    global_counts = run_parallel_pretokenization(input_path, special_tokens)

    vocab = {i: bytes([i]) for i in range(256)}
    vocab_idx = 256
    merges = []
    
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small to fit base bytes and special tokens.")

    splits = {word: [bytes([b]) for b in word] for word in global_counts}
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    print("构建初始字节对缓存与优先队列...")
    for word, split in splits.items():
        freq = global_counts[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # 🚀 优化核心 1：初始化大顶堆
    heap = [HeapItem(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    print(f"开始执行合并 (目标 {num_merges} 次)...")
    for i in range(num_merges):
        
        # 🚀 优化核心 2：O(1) 寻找最高频 pair (采用延迟验证机制)
        best_pair = None
        while heap:
            top_item = heapq.heappop(heap)
            # 由于更新时我们没删旧的堆元素，所以 pop 出来后必须验证 count 是否还是最新的
            if pair_counts.get(top_item.pair) == top_item.count:
                best_pair = top_item.pair
                break
                
        if not best_pair:
            break  # 已经没有任何可以合并的 pair 了

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        
        vocab[vocab_idx] = new_token
        vocab_idx += 1

        words_to_update = list(pair_to_words[best_pair])
        
        # 从统计中彻底移除
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_update:
            split = splits[word]
            freq = global_counts[word]

            # 步骤 A：扣除旧 pair
            for j in range(len(split) - 1):
                p = (split[j], split[j+1])
                if p != best_pair:
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    else:
                        # 🚀 计数改变，向堆里压入新状态
                        heapq.heappush(heap, HeapItem(pair_counts[p], p))
                    pair_to_words[p].discard(word)

            # 步骤 B：执行序列合并
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

            # 步骤 C：加上新生成的 pair
            for j in range(len(new_split) - 1):
                p = (new_split[j], new_split[j+1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)
                # 🚀 计数增加，向堆里压入新状态
                heapq.heappush(heap, HeapItem(pair_counts[p], p))
                
        if (i + 1) % 1000 == 0:
            print(f"  已完成合并: {i + 1} / {num_merges}")

    for st in special_tokens:
        vocab[vocab_idx] = st.encode("utf-8")
        vocab_idx += 1

    print("BPE 训练完成！")
    return vocab, merges


# ==========================================
# 测试/执行入口 (保持不变)
# ==========================================
if __name__ == "__main__":
    import time
    import pickle
    
    file_path = "/home/inferior/Projects/CS336/data/owt_train.txt"
    target_vocab_size = 32000
    special_tks = ["<|endoftext|>"]
    
    print("--- 开始 BPE 训练任务 ---")
    start_time = time.time()
    
    vocab, merges = train_bpe(
        input_path=file_path, 
        vocab_size=target_vocab_size, 
        special_tokens=special_tks
    )
    
    end_time = time.time()
    print(f"\n✅ 训练总耗时: {end_time - start_time:.2f} 秒")
    # ... 后续保存与打印逻辑不变 ...
    print(f"最终合并列表长度: {len(merges)}")
    print(f"最终词表大小: {len(vocab)}")
    
    # ---------------------------------------------------------
    # 序列化保存 (Serialization)
    # ---------------------------------------------------------
    output_dir = "tokenizer_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    merges_path = os.path.join(output_dir, "merges.pkl")
    
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
        
    print(f"💾 序列化完成！文件已保存至: {output_dir}/")
    
    # ---------------------------------------------------------
    # 分析：寻找最长的 Token (作业要求)
    # ---------------------------------------------------------
    # 找出 vocab 中 bytes 长度最长的那个 token
    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    
    print("\n🔍 词表分析:")
    print(f"最长 Token 的 ID: {longest_token_id}")
    print(f"最长 Token 的字节长度: {len(longest_token_bytes)}")
    try:
        # 尝试解码为人类可读的字符串
        print(f"最长 Token 的文本内容: {longest_token_bytes.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print(f"最长 Token (原始字节): {longest_token_bytes}")

    # 打印前 5 个发生的合并看看效果
    print("\n最初的 5 次 Merge:")
    for i in range(min(5, len(merges))):
        p1, p2 = merges[i]
        print(f"  {p1} + {p2} -> {p1 + p2}")