import pickle
import regex as re
from typing import Iterable, Iterator

# ==========================================
# 1. 全局配置与正则定义
# ==========================================
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_REGEX = re.compile(PATTERN)

# ==========================================
# 2. Tokenizer 核心类
# ==========================================
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化 Tokenizer，建立查找索引并处理特殊 Token。
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 建立反向查找字典，用于快速获取 token 对应的 ID
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        
        # 将 merges 列表转化为 {pair: rank} 字典，以 O(1) 的时间复杂度获取合并优先级
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        # 动态处理传入的 special_tokens：如果不在词表中，则追加到末尾
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.token_to_id:
                self.vocab[next_id] = st_bytes
                self.token_to_id[st_bytes] = next_id
                next_id += 1

        # 编译用于切分特殊 Token 的正则表达式
        self.st_pattern = None
        if self.special_tokens:
            # 使用捕获组 (...)，这样 re.split 时能保留特殊 token 在列表中
            sorted_st = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(map(re.escape, sorted_st))
            self.st_pattern = re.compile(f"({pattern_str})")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        从序列化的 pickle 文件中加载 vocab 和 merges 并实例化 Tokenizer。
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_chunk(self, chunk_text: str) -> list[int]:
        """
        内部辅助函数：对单个正则切割后的纯文本 pre-token 进行 BPE 合并操作。
        """
        # 将字符串打散成单字节列表
        b_list = [bytes([b]) for b in chunk_text.encode("utf-8")]
        
        # 如果长度为1，直接返回（基础256字节必然在词表中）
        if len(b_list) <= 1:
            return [self.token_to_id[b] for b in b_list]

        while True:
            # 找到当前序列中，在 merges 中优先级最高（rank 最小）的相邻字节对
            min_rank = float('inf')
            best_pair = None
            
            for i in range(len(b_list) - 1):
                pair = (b_list[i], b_list[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            # 如果没有任何一对在 merge_ranks 中，说明合并彻底完成
            if best_pair is None:
                break

            # 执行最高优先级的一轮合并
            new_b_list = []
            i = 0
            while i < len(b_list):
                # 命中目标 pair
                if i < len(b_list) - 1 and b_list[i] == best_pair[0] and b_list[i+1] == best_pair[1]:
                    new_b_list.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_b_list.append(b_list[i])
                    i += 1
                    
            b_list = new_b_list
            
            # 如果只剩下一个完整的 token，提前退出
            if len(b_list) == 1:
                break

        # 拿着最终合并好的字节序列，去词表里换取对应的 IDs
        return [self.token_to_id[b] for b in b_list]

    def encode(self, text: str) -> list[int]:
        """
        将整段输入文本编码为 token IDs。
        """
        ids = []
        
        # 1. 全局截断：优先将文本通过 special_tokens 切开
        if self.st_pattern:
            parts = self.st_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
                
            # 2. 如果这块 part 是一个 special token，直接查 ID
            if part in self.special_tokens:
                ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                # 3. 对普通文本进行预分词匹配
                for match in PRETOKEN_REGEX.finditer(part):
                    chunk_text = match.group()
                    ids.extend(self._bpe_encode_chunk(chunk_text))
                    
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        懒加载生成器：逐块/逐行读取流数据并输出 IDs，确保内存开销维持在常数级别。
        说明：对于标准的文件句柄 (File Handle)，每次 yield 的是一行。
        换行符属于 BPE 空格家族，因此按行拆分不会截断合法的词组边界。
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """
        将整数 ID 序列还原为字符串。非法字节将被自动替换为 U+FFFD ()。
        """
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")