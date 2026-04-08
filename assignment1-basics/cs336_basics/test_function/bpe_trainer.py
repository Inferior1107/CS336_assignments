import os
import time
import pickle
import regex as re
import multiprocessing
from typing import BinaryIO
from collections import Counter

import fast_bpe

# ==========================================
# 1. 全局配置与正则定义
# ==========================================

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_REGEX = re.compile(PATTERN)

# ==========================================
# 2. 多进程预分词辅助函数 (保持原样，Python 的强项)
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
    print("开始预分词与频率统计 (Python 多进程)...")
    global_counts = run_parallel_pretokenization(input_path, special_tokens)

    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small to fit base bytes and special tokens.")

    print(f"将数据移交 Rust 底层进行合并 (目标总词表大小: {vocab_size - len(special_tokens)})...")
    
    merges, vocab = fast_bpe.train_bpe_rust(global_counts, vocab_size - len(special_tokens))

    print("Rust 计算完成，添加 Special Tokens...")
    
    vocab_idx = 256 + len(merges)
    for st in special_tokens:
        vocab[vocab_idx] = st.encode("utf-8")
        vocab_idx += 1

    print("BPE 训练彻底完成！")
    return vocab, merges


# ==========================================
# 测试/执行入口
# ==========================================
if __name__ == "__main__":
    
    # file_path = "/home/inferior/Projects/CS336/data/owt_train.txt"
    file_path = "/home/inferior/Projects/CS336/data/TinyStoriesV2-GPT4-train.txt"
    # target_vocab_size = 32000
    target_vocab_size = 10000
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
    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    
    print("\n🔍 词表分析:")
    print(f"最长 Token 的 ID: {longest_token_id}")
    print(f"最长 Token 的字节长度: {len(longest_token_bytes)}")
    try:
        print(f"最长 Token 的文本内容: {longest_token_bytes.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print(f"最长 Token (原始字节): {longest_token_bytes}")

    print("\n最初的 5 次 Merge:")
    for i in range(min(5, len(merges))):
        p1, p2 = merges[i]
        print(f"  {p1} + {p2} -> {p1 + p2}")