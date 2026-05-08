
import os
import json
import torch


def get_bpe_file(root_path):
    all_matchs = []
    for file_name in os.listdir(root_path):
        if file_name.endswith('tiktoken'):
            all_matchs.append(os.path.join(root_path, file_name))
    assert len(all_matchs) == 1, f"Multiple / No tiktoken bpe files found: {all_matchs}"
    return all_matchs[0]


def _gpt2_bytes_to_unicode():
    """
    Maps each of the 256 byte values to a unique printable Unicode char.
    This is the same table used by GPT-2 / Qwen BPE tokenizers.
    Returns dict: byte_int -> unicode_char_int
    """
    bs = (
        list(range(ord("!"),  ord("~")  + 1))   # 33-126
        + list(range(ord("¡"), ord("¬") + 1))   # 161-172
        + list(range(ord("®"), ord("ÿ") + 1))   # 174-255
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))   # byte_int -> char_int


def build_bytes_list(tokenizer_dir, vocab_size):
    """
    Build old_bytes_list[i] = raw bytes for token id i, using the HF BPE
    tokenizer at tokenizer_dir (GPT-2 byte-level encoded vocab).

      - BPE vocab tokens   : decoded via GPT-2 byte table.
      - added_tokens        : raw UTF-8 of their content string.
      - Padding slots       : unique dummy bytes (zero counts, always dropped).
    """
    tok_json_path = os.path.join(tokenizer_dir, 'tokenizer.json')
    with open(tok_json_path, 'r', encoding='utf-8') as f:
        tok_json = json.load(f)

    byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}  # char_int -> byte_int

    bpe_vocab    = tok_json['model']['vocab']                              # token_str -> id
    added_tokens = {at['id']: at['content']
                    for at in tok_json.get('added_tokens', [])}

    id_to_bytes = {}
    for token_str, token_id in bpe_vocab.items():
        raw = bytes(byte_decoder.get(ord(c), ord(c) % 256) for c in token_str)
        id_to_bytes[token_id] = raw
    for token_id, content in added_tokens.items():
        id_to_bytes[token_id] = content.encode('utf-8')

    result = []
    for i in range(vocab_size):
        if i in id_to_bytes:
            result.append(id_to_bytes[i])
        else:
            # Padding slot — unique bytes that will never match any BPE sub-token
            result.append(b'\x00\xff' + i.to_bytes(3, 'big'))
    return result


def make_context(
    tokenizer,
    query,
    history=None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    """Legacy Qwen 1.x chat context builder — kept for backward compatibility."""
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            if hasattr(tokenizer, 'IMAGE_ST'):
                return f"{role}\n{content}", tokenizer.encode(
                    role, allowed_special=set(tokenizer.IMAGE_ST)
                ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
            else:
                return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens
