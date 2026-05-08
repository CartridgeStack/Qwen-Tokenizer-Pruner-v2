import os
from tqdm import tqdm
import json
import torch


def get_text_list(folder_path):
    query_list = []
    prompt_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('json'):
            file = json.load(open(os.path.join(folder_path, file_name),
                                  encoding='utf-8'))
            if 'query' in file:
                query_list.append(file['query'])
            if 'response' in file:
                prompt_list.append(file['response'])
            if 'prompt' in file:
                prompt_list.append(file['prompt'])
    return query_list, prompt_list


def count_freq(data_path, vocab_size, tokenizer, output_path, inherit_vocab_count):
    vocab_counts = [0 for _ in range(vocab_size)]
    query_list, prompt_list = get_text_list(data_path)

    print("calculate query vocab counts: apply chat template then encode")
    for i in tqdm(range(len(query_list))):
        query = query_list[i]
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": query},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = query
        for token in tokenizer.encode(text):
            if token < vocab_size:
                vocab_counts[token] += 1

    print("calculate prompt vocab counts: encode directly")
    for i in tqdm(range(len(prompt_list))):
        for token in tokenizer.encode(prompt_list[i]):
            if token < vocab_size:
                vocab_counts[token] += 1

    if inherit_vocab_count is not None:
        if os.path.exists(inherit_vocab_count):
            print(f"==> Load inherit_vocab_count from: {inherit_vocab_count}")
            inherited = torch.load(inherit_vocab_count)
            assert len(inherited) == vocab_size
            for token, cnt in enumerate(inherited):
                vocab_counts[token] += int(cnt)
        else:
            print("==> No valid inherit_vocab_count path, skipping.")

    torch.save(vocab_counts, os.path.join(output_path, 'vocab_counts.torch'))
    return vocab_counts


def is_special_token(token):
    return ((token.startswith('<') and token.endswith('>') and len(token) > 2) or
            (token.startswith('[') and token.endswith(']') and len(token) > 2))


def update_vocab_count_by_langfilter(support_lang, vocab_counts, old_bytes_list, count_offset=1):
    from langdetect import detect as langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0

    for i in tqdm(range(len(old_bytes_list))):
        token_bytes = old_bytes_list[i]
        try:
            token_str = token_bytes.decode("utf-8")
            if (langdetect(token_str) in support_lang) or is_special_token(token_str):
                vocab_counts[i] += count_offset
        except Exception:
            vocab_counts[i] += count_offset
    return vocab_counts


def count_recursive(vocab_size, vocab_counts, old_bytes_list):
    recursive_counts = [0 for _ in range(vocab_size)]

    # O(N) lookup dict instead of list.index (was O(N²) per token)
    bytes_to_idx = {}
    for i, b in enumerate(old_bytes_list):
        if b not in bytes_to_idx:
            bytes_to_idx[b] = i

    for i in tqdm(range(len(old_bytes_list))):
        token_bytes = old_bytes_list[i]
        t_count = vocab_counts[i]
        b_len = len(token_bytes)
        if t_count > 0 and b_len > 1:
            for j in range(1, b_len):
                for k in range(b_len + 1 - j):
                    sub_token = token_bytes[k:j + k]
                    idx = bytes_to_idx.get(sub_token)
                    if idx is not None:
                        recursive_counts[idx] += t_count

    return recursive_counts
