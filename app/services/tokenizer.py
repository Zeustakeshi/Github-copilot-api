"""
Tokenizer service — đếm tokens bằng tiktoken (offline, không cần gọi API).
"""

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Đếm token bằng tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
