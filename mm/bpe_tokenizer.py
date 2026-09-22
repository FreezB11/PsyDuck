"""
Python reader/encoder for the custom BPE2 tokenizer.bin format
produced by tokenizer.c.

File format (little-endian):
    magic[4]        = "BPE2"
    version  uint32
    vocab_size uint32
    merge_count uint32
    reserved uint32
    vocab_size * Token{ left uint32, right uint32 }
    merge_count * pair_t (uint64, packed as (left<<32)|right)

Token ids 0..255 are raw bytes. Merge i produces token id 256+i,
matching add_token() in tokenizer.c.

Encode uses a heap-based merge (O(n log n)) instead of the naive
full-rescan-per-merge approach, so it stays fast on large corpora.
"""

import struct
import heapq

NONE = -1


class BPETokenizer:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = f.read()

        magic = data[0:4]
        if magic != b"BPE2":
            raise ValueError(f"bad magic: {magic!r}, expected b'BPE2'")

        version, vocab_size, merge_count, _reserved = struct.unpack_from(
            "<IIII", data, 4
        )
        offset = 4 + 16

        vocab = []
        for _ in range(vocab_size):
            left, right = struct.unpack_from("<II", data, offset)
            vocab.append((left, right))
            offset += 8

        merges = []
        for _ in range(merge_count):
            (packed,) = struct.unpack_from("<Q", data, offset)
            merges.append(packed)
            offset += 8

        self.version = version
        self.vocab = vocab                 # id -> (left, right)
        self.merges = merges               # merge order -> packed pair
        self.vocab_size = vocab_size       # size BEFORE adding EOS
        # Reserve one id past the trained vocab for EOS. BPE will never
        # emit this id on its own, so it's safe as a sentinel.
        self.eos_token_id = vocab_size
        self.total_vocab_size = vocab_size + 1  # what MODEL_CONFIG should use

        # pair (as packed uint64 key) -> merge rank (lower = higher priority)
        self.rank = {p: i for i, p in enumerate(merges)}

        print(
            f"Loaded tokenizer: vocab={vocab_size} merges={merge_count} "
            f"(eos_token_id={self.eos_token_id}, total_vocab_size={self.total_vocab_size})"
        )

    @staticmethod
    def _pair_key(a: int, b: int) -> int:
        return (a << 32) | b

    def encode(self, text) -> list[int]:
        """Encode a str or bytes object into a list of token ids."""
        if isinstance(text, str):
            data = text.encode("utf-8")
        else:
            data = text

        n = len(data)
        if n == 0:
            return []

        tokens = list(data)                      # array position -> current token id
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = NONE
        alive = [True] * n

        rank = self.rank
        pair_key = self._pair_key
        heap = []

        # seed heap with all initial adjacent pairs
        i = 0
        while i != NONE:
            j = nxt[i]
            if j == NONE:
                break
            r = rank.get(pair_key(tokens[i], tokens[j]))
            if r is not None:
                heapq.heappush(heap, (r, i))
            i = j

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == NONE:
                continue

            # confirm this entry isn't stale (pair may have changed since pushed)
            key = pair_key(tokens[i], tokens[j])
            cur_rank = rank.get(key)
            if cur_rank != r:
                continue

            new_token = 256 + r
            p = prev[i]
            nn = nxt[j]

            tokens[i] = new_token
            nxt[i] = nn
            if nn != NONE:
                prev[nn] = i
            alive[j] = False

            if p != NONE:
                rl = rank.get(pair_key(tokens[p], tokens[i]))
                if rl is not None:
                    heapq.heappush(heap, (rl, p))
            if nn != NONE:
                rr = rank.get(pair_key(tokens[i], tokens[nn]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i))

        out = []
        i = 0
        while i != NONE:
            out.append(tokens[i])
            i = nxt[i]
        return out

    def decode(self, ids) -> bytes:
        out = bytearray()

        def expand(tid: int):
            if tid < 256:
                out.append(tid)
                return
            if tid == self.eos_token_id or tid >= len(self.vocab):
                return  # skip EOS / unknown ids
            left, right = self.vocab[tid]
            expand(left)
            expand(right)

        for tid in ids:
            expand(tid)
        return bytes(out)


# ----------------------------------------------------------------
# Drop-in replacement for tokenize_all() in your training script
# ----------------------------------------------------------------
def tokenize_all(dataset, tokenizer: BPETokenizer):
    """
    Mirrors the tiktoken-based tokenize_all() in your training script,
    but using this custom tokenizer + its reserved EOS id.
    """
    all_ids = []
    for text in dataset["text"]:
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)
        all_ids.extend(ids)
    return all_ids


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bpe_tokenizer.py <tokenizer.bin> [text to encode]")
        sys.exit(1)

    tok = BPETokenizer(sys.argv[1])
    sample = " ".join(sys.argv[2:]) or "Question: If 3x + 5 = 20, what is x?\nAnswer:"
    ids = tok.encode(sample)
    print(f"\nInput:  {sample!r}")
    print(f"Tokens: {ids}")
    print(f"Count:  {len(ids)} tokens for {len(sample.encode('utf-8'))} bytes "
          f"({len(sample.encode('utf-8')) / max(len(ids), 1):.2f} bytes/token)")
    print(f"Decoded: {tok.decode(ids)!r}")
