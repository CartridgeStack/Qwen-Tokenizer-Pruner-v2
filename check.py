import os
import json
import torch
import argparse
from transformers import AutoTokenizer
from vocab_count import get_text_list
from tqdm import tqdm


def main():
    print('============ Lossless Check: Old vs New Tokenizer ==========')

    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--support_data',   type=str, required=True)
    args = parser.parse_args()

    print(f"Loading old tokenizer from {args.old_model_path}")
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    print(f"Loading new tokenizer from {args.new_model_path}")
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_model_path, trust_remote_code=True)

    mapping_file = os.path.join(args.new_model_path, 'token_mapping.torch')
    print(f"Loading token mapping from {mapping_file}")
    mapping_new2old = torch.load(mapping_file).long().tolist()

    query_list, prompt_list = get_text_list(args.support_data)
    all_texts = []

    # Build check texts with chat formatting where available
    for query in query_list:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": query},
            ]
            text = old_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = query
        all_texts.append(text)

    for prompt in prompt_list:
        all_texts.append(prompt)

    print(f"Checking {len(all_texts)} texts...")
    mismatches = []
    for text in tqdm(all_texts):
        old_tokens = old_tokenizer.encode(text)
        new_tokens = new_tokenizer.encode(text)
        if len(old_tokens) != len(new_tokens):
            mismatches.append(text)
            continue
        if not all(old_t == mapping_new2old[new_t]
                   for old_t, new_t in zip(old_tokens, new_tokens)):
            mismatches.append(text)

    if mismatches:
        print(f"==> MISMATCHES: {len(mismatches)} / {len(all_texts)}")
        sample = old_tokenizer.encode(mismatches[0])
        print(f"  Example old tokens[:10]: {sample[:10]}")
        sample_new = new_tokenizer.encode(mismatches[0])
        print(f"  Example new tokens[:10]: {sample_new[:10]}")
        print(f"  Mapped back:             {[mapping_new2old[t] for t in sample_new[:10]]}")
    else:
        print(f"==> All {len(all_texts)} texts encode identically. Lossless check PASSED.")


if __name__ == '__main__':
    main()
