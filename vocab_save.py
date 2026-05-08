import os
import json
import base64
import torch
from tqdm import tqdm


def reduce_to_target_size(old_vocab_size, target_vocab_size, vocab_counts, recur_counts, old_bytes_list):
    total_count_with_idx = [(vocab_counts[i] + recur_counts[i], i) for i in range(old_vocab_size)]
    sorted_count_with_idx = sorted(total_count_with_idx, key=lambda x: x[0])
    remove_count = 0
    remove_target = old_vocab_size - target_vocab_size

    for i in tqdm(range(len(sorted_count_with_idx))):
        token_count, token_idx = sorted_count_with_idx[i]
        if remove_count >= remove_target:
            continue
        elif token_count == 0:
            remove_count += 1
        elif len(old_bytes_list[token_idx]) > 1:
            token = old_bytes_list[token_idx]
            b_len = len(token)
            for j in range(1, b_len):
                if (token[:j] in old_bytes_list) and (token[j:] in old_bytes_list):
                    parta_index = old_bytes_list.index(token[:j])
                    partb_index = old_bytes_list.index(token[j:])
                    if (vocab_counts[parta_index] + recur_counts[parta_index] > 0) and \
                       (vocab_counts[partb_index] + recur_counts[partb_index] > 0):
                        vocab_counts[token_idx] = 0
                        recur_counts[token_idx] = 0
                        remove_count += 1
                        break

    if remove_count < remove_target:
        print(f"Failed to reach the target size")
    return vocab_counts, recur_counts


def get_new_vocab_and_map(old_bytes_list, old_vocab_size, vocab_counts, recur_counts):
    new_bytes_list = []
    mapping_new2old = []

    for i in tqdm(range(len(old_bytes_list))):
        if vocab_counts[i] + recur_counts[i] > 0:
            new_bytes_list.append(old_bytes_list[i])
            mapping_new2old.append(i)

    # For tiktoken-style models the special tokens live beyond len(old_bytes_list);
    # for HF models they are already in old_bytes_list with forced counts, so
    # this range will be empty (len == old_vocab_size).
    extra = old_vocab_size - len(old_bytes_list)
    if extra > 0:
        print(f"Add special token (num: {extra})")
        for i in range(len(old_bytes_list), old_vocab_size):
            mapping_new2old.append(i)

    print(f"Vocabulary size: {old_vocab_size} => {len(mapping_new2old)}")
    return new_bytes_list, mapping_new2old


def save_vocab(bytes_list, token_mapping, output_path, old_model_path=None):
    """
    Save the pruned vocabulary.

    For HF BPE models (old_model_path provided):
      Writes tokenizer.json, vocab.json, merges.txt into output_path,
      and patches tokenizer_config.json's added_tokens_decoder.

    Legacy fallback (old_model_path=None):
      Writes a qwen.tiktoken file (Qwen 1.x format).

    Always writes token_mapping.torch.
    """
    token_mapping_path = os.path.join(output_path, 'token_mapping.torch')
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
    print(f"Mapping file (new token 2 old token) saved: {token_mapping_path}")

    if old_model_path is None:
        # ── Legacy tiktoken path ──────────────────────────────────────────────
        new_tiktoken_path = os.path.join(output_path, 'qwen.tiktoken')
        with open(new_tiktoken_path, "w", encoding="utf8") as w:
            for i, token in enumerate(bytes_list):
                line = base64.b64encode(token).decode("utf8") + " " + str(i) + "\n"
                w.write(line)
        print(f"New tiktoken BPE file (size: {len(bytes_list)}) saved: {new_tiktoken_path}")
        return

    # ── HF BPE tokenizer rebuild ──────────────────────────────────────────────
    tok_json_path = os.path.join(old_model_path, 'tokenizer.json')
    with open(tok_json_path, 'r', encoding='utf-8') as f:
        tok_json = json.load(f)

    old_bpe_vocab    = tok_json['model']['vocab']           # token_str -> old_id
    old_merges       = tok_json['model']['merges']          # ["A B", ...]
    old_added_tokens = tok_json.get('added_tokens', [])

    # old_id -> new_id for every kept token
    new_id_from_old = {old_id: new_id for new_id, old_id in enumerate(token_mapping)}

    # ── 1. New BPE vocab (kept tokens only, renumbered) ──────────────────────
    new_bpe_vocab = {}
    for token_str, old_id in old_bpe_vocab.items():
        if old_id in new_id_from_old:
            new_bpe_vocab[token_str] = new_id_from_old[old_id]

    # Full vocab including added_tokens (used for merge filtering)
    full_new_vocab = dict(new_bpe_vocab)
    for at in old_added_tokens:
        if at['id'] in new_id_from_old:
            full_new_vocab[at['content']] = new_id_from_old[at['id']]

    # ── 2. Filter merges ──────────────────────────────────────────────────────
    # Keep "A B" only if A, B, and their concatenation A+B are all in the new vocab.
    new_merges = []
    for merge in old_merges:
        parts = merge.split(' ', 1)
        if len(parts) != 2:
            continue
        a, b = parts
        if a in full_new_vocab and b in full_new_vocab and (a + b) in full_new_vocab:
            new_merges.append(merge)
    print(f"Merges: {len(old_merges)} -> {len(new_merges)}")

    # ── 3. New added_tokens list ──────────────────────────────────────────────
    new_added_tokens = []
    for at in old_added_tokens:
        if at['id'] in new_id_from_old:
            updated = dict(at)
            updated['id'] = new_id_from_old[at['id']]
            new_added_tokens.append(updated)
    new_added_tokens.sort(key=lambda x: x['id'])

    # ── 4. Write tokenizer.json ───────────────────────────────────────────────
    new_tok_json = dict(tok_json)
    new_tok_json['added_tokens'] = new_added_tokens
    new_tok_json['model'] = dict(tok_json['model'])
    new_tok_json['model']['vocab'] = new_bpe_vocab
    new_tok_json['model']['merges'] = new_merges

    out_tok = os.path.join(output_path, 'tokenizer.json')
    with open(out_tok, 'w', encoding='utf-8') as f:
        json.dump(new_tok_json, f, indent=2, ensure_ascii=False)
    print(f"tokenizer.json: {len(new_bpe_vocab)} vocab entries, {len(new_merges)} merges")

    # ── 5. Write vocab.json ───────────────────────────────────────────────────
    out_vocab = os.path.join(output_path, 'vocab.json')
    with open(out_vocab, 'w', encoding='utf-8') as f:
        json.dump(new_bpe_vocab, f, indent=2, ensure_ascii=False)
    print(f"vocab.json written")

    # ── 6. Write merges.txt ───────────────────────────────────────────────────
    out_merges = os.path.join(output_path, 'merges.txt')
    with open(out_merges, 'w', encoding='utf-8') as f:
        f.write('#version: 0.2\n')
        for merge in new_merges:
            f.write(merge + '\n')
    print(f"merges.txt written")

    # ── 7. Patch tokenizer_config.json added_tokens_decoder ──────────────────
    tok_cfg_path = os.path.join(output_path, 'tokenizer_config.json')
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, 'r', encoding='utf-8') as f:
            tok_cfg = json.load(f)
        if 'added_tokens_decoder' in tok_cfg:
            new_atd = {}
            for old_id_str, info in tok_cfg['added_tokens_decoder'].items():
                old_id = int(old_id_str)
                if old_id in new_id_from_old:
                    new_atd[str(new_id_from_old[old_id])] = info
            tok_cfg['added_tokens_decoder'] = new_atd
        with open(tok_cfg_path, 'w', encoding='utf-8') as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
        print(f"tokenizer_config.json: patched added_tokens_decoder")
