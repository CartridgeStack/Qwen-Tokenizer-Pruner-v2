import os
import json
import torch
import argparse
from transformers import AutoTokenizer
from vocab_count import count_freq, update_vocab_count_by_langfilter, count_recursive
from vocab_save import get_new_vocab_and_map, save_vocab, reduce_to_target_size
from model_save import saving_updated_qwen_lowmem
from utils import build_bytes_list
from tqdm import tqdm

from langdetect import DetectorFactory
DetectorFactory.seed = 0


def main():
    print('============ Start Qwen Vocabulary Pruning ==========')

    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--support_data', type=str, default=None)
    parser.add_argument('--support_lang', default=[], type=str, nargs='+')
    parser.add_argument('--inherit_vocab_count', type=str, default=None)
    parser.add_argument('--target_size', type=int, default=None)
    args = parser.parse_args()

    assert (args.support_data is not None) or (len(args.support_lang) > 0), \
        "Must provide at least one pruning method: --support_data or --support_lang."

    if not os.path.exists(args.new_model_path):
        os.makedirs(args.new_model_path)
        print(f"==> Created output folder: {args.new_model_path}")

    # ── Read vocab_size from config.json directly (handles nested text_config) ──
    cfg_json_path = os.path.join(args.old_model_path, 'config.json')
    with open(cfg_json_path, 'r', encoding='utf-8') as f:
        raw_cfg = json.load(f)
    old_vocab_size = (
        raw_cfg.get('vocab_size')
        or raw_cfg.get('text_config', {}).get('vocab_size')
    )
    assert old_vocab_size is not None, \
        "Could not find vocab_size in config.json (checked top-level and text_config)."
    print(f"==> vocab_size: {old_vocab_size}")

    # ── Load tokenizer only — no full model load needed ──────────────────────
    print(f"==> Loading tokenizer from: {args.old_model_path}")
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)

    # ── Count token frequencies from support data ─────────────────────────────
    if args.support_data is not None:
        print(f"==> Counting token frequencies from: {args.support_data}")
        vocab_counts = count_freq(
            data_path=args.support_data,
            vocab_size=old_vocab_size,
            tokenizer=old_tokenizer,
            output_path=args.new_model_path,
            inherit_vocab_count=args.inherit_vocab_count,
        )
    else:
        vocab_counts = [0] * old_vocab_size

    # ── Force-keep all special / added tokens ────────────────────────────────
    tok_json_path = os.path.join(args.old_model_path, 'tokenizer.json')
    with open(tok_json_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
    forced = 0
    for at in tok_data.get('added_tokens', []):
        tid = at['id']
        if tid < old_vocab_size and vocab_counts[tid] == 0:
            vocab_counts[tid] = 1
            forced += 1
    print(f"==> Force-kept {forced} added/special tokens with zero frequency")

    # ── Build byte representation of every token (needed for sub-token logic) ─
    print(f"==> Building byte list from HF BPE tokenizer")
    old_bytes_list = build_bytes_list(args.old_model_path, old_vocab_size)

    # ── Optional: language filter ─────────────────────────────────────────────
    if len(args.support_lang) > 0:
        print(f"==> Filtering by language: {args.support_lang}")
        vocab_counts = update_vocab_count_by_langfilter(
            support_lang=args.support_lang,
            vocab_counts=vocab_counts,
            old_bytes_list=old_bytes_list,
            count_offset=1,
        )

    # ── Recursive sub-token count (ensures BPE building blocks are kept) ──────
    print(f"==> Calculating recursive sub-token counts")
    recur_counts = count_recursive(
        vocab_size=old_vocab_size,
        vocab_counts=vocab_counts,
        old_bytes_list=old_bytes_list,
    )

    # ── Optional: hard cap on vocab size ─────────────────────────────────────
    if args.target_size is not None:
        print(f"==> Reducing vocab to target size {args.target_size}")
        vocab_counts, recur_counts = reduce_to_target_size(
            old_vocab_size=old_vocab_size,
            target_vocab_size=args.target_size,
            vocab_counts=vocab_counts,
            recur_counts=recur_counts,
            old_bytes_list=old_bytes_list,
        )

    # ── Build new vocab and token mapping ────────────────────────────────────
    print(f"==> Building new vocabulary and mapping")
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(
        old_bytes_list=old_bytes_list,
        old_vocab_size=old_vocab_size,
        vocab_counts=vocab_counts,
        recur_counts=recur_counts,
    )
    new_vocab_size = len(mapping_new2old)

    # ── Patch model weights shard-by-shard (low-memory) ──────────────────────
    # This also copies all non-safetensors files (configs, tokenizer files, etc.)
    print(f"==> Patching model weights (shard-by-shard, low-memory)")
    saving_updated_qwen_lowmem(
        old_model_path=args.old_model_path,
        new_vocab_size=new_vocab_size,
        mapping_new2old=mapping_new2old,
        output_path=args.new_model_path,
        old_vocab_size=old_vocab_size,
    )

    # ── Write new tokenizer files (overwrites the copied originals) ───────────
    print(f"==> Writing new HF BPE tokenizer files")
    save_vocab(
        bytes_list=new_bytes_list,
        token_mapping=mapping_new2old,
        output_path=args.new_model_path,
        old_model_path=args.old_model_path,
    )

    print(f"\n==> Done. Pruned model at: {args.new_model_path}")
    print(f"    Vocab: {old_vocab_size} -> {new_vocab_size} tokens")


if __name__ == '__main__':
    main()
